package com.detectorpreventor.app.notifications

import android.content.ComponentName
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.Settings
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.AudioSpectrogramGenerator
import com.detectorpreventor.app.ml.TfliteExecutionLock
import com.detectorpreventor.app.ml.VisionClassifier
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Collections
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class InterceptedNotification(
    val id: String,
    val appName: String,
    val packageName: String,
    val sender: String,
    val text: String,
    val timestamp: Long,
    val mediaType: MediaType,
    val riskScore: Float,
    val isThreat: Boolean,
    val fusionResult: FusionResult? = null,
    val mediaUriString: String? = null
) {
    val formattedTime: String
        get() = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(timestamp))

    fun isFaceDiscarded(): Boolean = fusionResult?.wasFaceDiscarded == true

    fun isPendingDownload(): Boolean {
        if (isFaceDiscarded()) return false
        return (riskScore == 0.0f && fusionResult == null) ||
                text.contains("descarga", ignoreCase = true) ||
                text.contains("pendiente", ignoreCase = true) ||
                text.contains("descargando", ignoreCase = true)
    }

    fun toJson(): JSONObject {
        val obj = JSONObject()
        obj.put("id", id)
        obj.put("appName", appName)
        obj.put("packageName", packageName)
        obj.put("sender", sender)
        obj.put("text", text)
        obj.put("timestamp", timestamp)
        obj.put("mediaType", mediaType.name)
        obj.put("riskScore", riskScore.toDouble())
        obj.put("isThreat", isThreat)
        mediaUriString?.let { obj.put("mediaUriString", it) }
        fusionResult?.let { fr ->
            val frObj = JSONObject()
            frObj.put("globalRiskPercentage", fr.globalRiskPercentage.toDouble())
            frObj.put("audioFraudProb", fr.audioFraudProb.toDouble())
            frObj.put("visionFraudProb", fr.visionFraudProb.toDouble())
            frObj.put("riskLevel", fr.riskLevel.name)
            frObj.put("weightAudio", fr.weightAudio.toDouble())
            frObj.put("weightVision", fr.weightVision.toDouble())
            frObj.put("diagnosticSummary", fr.diagnosticSummary)
            frObj.put("wasFaceDiscarded", fr.wasFaceDiscarded)
            obj.put("fusionResult", frObj)
        }
        return obj
    }

    companion object {
        fun fromJson(obj: JSONObject): InterceptedNotification {
            val fr = if (obj.has("fusionResult") && !obj.isNull("fusionResult")) {
                val frObj = obj.getJSONObject("fusionResult")
                val rLevel = try {
                    RiskLevel.valueOf(frObj.optString("riskLevel", RiskLevel.BAJO.name))
                } catch (e: Exception) {
                    RiskLevel.BAJO
                }
                FusionResult(
                    globalRiskPercentage = frObj.optDouble("globalRiskPercentage", 0.0).toFloat(),
                    audioFraudProb = frObj.optDouble("audioFraudProb", 0.0).toFloat(),
                    visionFraudProb = frObj.optDouble("visionFraudProb", 0.0).toFloat(),
                    riskLevel = rLevel,
                    weightAudio = frObj.optDouble("weightAudio", 0.6).toFloat(),
                    weightVision = frObj.optDouble("weightVision", 0.4).toFloat(),
                    diagnosticSummary = frObj.optString("diagnosticSummary", ""),
                    wasFaceDiscarded = frObj.optBoolean("wasFaceDiscarded", false)
                )
            } else null

            val mType = try {
                MediaType.valueOf(obj.optString("mediaType", MediaType.UNKNOWN.name))
            } catch (e: Exception) {
                MediaType.UNKNOWN
            }

            val uriStr = if (obj.has("mediaUriString") && !obj.isNull("mediaUriString")) {
                obj.optString("mediaUriString", "").takeIf { it.isNotBlank() && it != "null" }
            } else null

            return InterceptedNotification(
                id = obj.optString("id", java.util.UUID.randomUUID().toString()),
                appName = obj.optString("appName", "WhatsApp"),
                packageName = obj.optString("packageName", "com.whatsapp"),
                sender = obj.optString("sender", "Desconocido"),
                text = obj.optString("text", ""),
                timestamp = obj.optLong("timestamp", System.currentTimeMillis()),
                mediaType = mType,
                riskScore = obj.optDouble("riskScore", 0.0).toFloat(),
                isThreat = obj.optBoolean("isThreat", false),
                fusionResult = fr,
                mediaUriString = uriStr
            )
        }
    }
}

object NotificationRepository {
    private const val PREFS_NAME = "detector_notifications_store"
    private const val KEY_NOTIFS = "saved_notifications"
    private const val KEY_BLOCKED_SENDERS = "blocked_senders_list"
    private const val MAX_SAVED_ITEMS = 50

    private var prefs: SharedPreferences? = null
    private val repoLock = Any()
    private val pendingResolveExecutor = Executors.newSingleThreadExecutor()
    private val isResolvingPending = AtomicBoolean(false)

    private val _notifications = MutableStateFlow<List<InterceptedNotification>>(emptyList())
    val notifications: StateFlow<List<InterceptedNotification>> = _notifications.asStateFlow()

    private val _blockedSenders = MutableStateFlow<Set<String>>(emptySet())
    val blockedSenders: StateFlow<Set<String>> = _blockedSenders.asStateFlow()

    private val _isMonitoringActive = MutableStateFlow(true)
    val isMonitoringActive: StateFlow<Boolean> = _isMonitoringActive.asStateFlow()

    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    fun init(context: Context) {
        val app = context.applicationContext
        synchronized(repoLock) {
            prefs = app.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            loadFromStorageLocked()
        }
        autoResolvePendingNotifications(app)
    }

    private fun loadFromStorageLocked() {
        val savedBlocked = prefs?.getStringSet(KEY_BLOCKED_SENDERS, null)
        if (savedBlocked != null) {
            _blockedSenders.value = savedBlocked.toSet()
        }

        val jsonString = prefs?.getString(KEY_NOTIFS, null) ?: return
        try {
            val jsonArray = JSONArray(jsonString)
            val list = mutableListOf<InterceptedNotification>()
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                list.add(InterceptedNotification.fromJson(obj))
            }
            _notifications.value = list
        } catch (_: Exception) {
        }
    }

    private fun persistToStorageLocked(list: List<InterceptedNotification>) {
        val p = prefs ?: return
        try {
            val jsonArray = JSONArray()
            list.forEach { jsonArray.put(it.toJson()) }
            p.edit().putString(KEY_NOTIFS, jsonArray.toString()).apply()
        } catch (_: Exception) {
        }
    }

    fun setMonitoringActive(active: Boolean) {
        _isMonitoringActive.value = active
    }

    fun setServiceConnected(connected: Boolean) {
        _isServiceConnected.value = connected
    }

    fun hasNotification(id: String): Boolean = synchronized(repoLock) {
        _notifications.value.any { it.id == id }
    }

    fun isSenderBlocked(sender: String?): Boolean {
        if (sender.isNullOrBlank()) return false
        val clean = sender.trim().lowercase()
        val blockedSnapshot = synchronized(repoLock) { _blockedSenders.value }
        return blockedSnapshot.any { blocked ->
            clean.equals(blocked, ignoreCase = true) ||
                clean.contains(blocked, ignoreCase = true) ||
                blocked.contains(clean, ignoreCase = true)
        }
    }

    fun blockSender(sender: String) {
        val clean = sender.trim().lowercase()
        if (clean.isBlank()) return
        synchronized(repoLock) {
            val current = _blockedSenders.value.toMutableSet()
            current.add(clean)
            val frozen = current.toSet()
            _blockedSenders.value = frozen
            prefs?.edit()?.putStringSet(KEY_BLOCKED_SENDERS, frozen)?.apply()
        }
    }

    fun unblockSender(sender: String) {
        val clean = sender.trim().lowercase()
        synchronized(repoLock) {
            val current = _blockedSenders.value.toMutableSet()
            current.removeAll { it.equals(clean, ignoreCase = true) || clean.contains(it, ignoreCase = true) }
            val frozen = current.toSet()
            _blockedSenders.value = frozen
            prefs?.edit()?.putStringSet(KEY_BLOCKED_SENDERS, frozen)?.apply()
        }
    }

    fun clearBlockedSenders() {
        synchronized(repoLock) {
            _blockedSenders.value = emptySet()
            prefs?.edit()?.remove(KEY_BLOCKED_SENDERS)?.apply()
        }
    }

    fun removeNotification(id: String): Boolean {
        var removedItem: InterceptedNotification? = null
        synchronized(repoLock) {
            val currentList = _notifications.value.toMutableList()
            val index = currentList.indexOfFirst { it.id == id }
            if (index != -1) {
                removedItem = currentList.removeAt(index)
                _notifications.value = currentList
                persistToStorageLocked(currentList)
            }
        }
        val removed = removedItem ?: return false
        removed.mediaUriString?.let { uriStr ->
            NotificationMediaCache.releaseFile(uriStr)
        }
        return true
    }

    fun removeNotificationByFilenameOrSender(query: String): Boolean {
        if (query.isBlank()) return false
        var removedItem: InterceptedNotification? = null
        synchronized(repoLock) {
            val currentList = _notifications.value.toMutableList()
            val index = currentList.indexOfFirst { item ->
                item.text.contains(query, ignoreCase = true) ||
                    item.sender.contains(query, ignoreCase = true) ||
                    (item.mediaUriString != null && item.mediaUriString.contains(query, ignoreCase = true))
            }
            if (index != -1) {
                removedItem = currentList.removeAt(index)
                _notifications.value = currentList
                persistToStorageLocked(currentList)
            }
        }
        val removed = removedItem ?: return false
        removed.mediaUriString?.let { uriStr ->
            NotificationMediaCache.releaseFile(uriStr)
        }
        return true
    }

    fun updateNotification(item: InterceptedNotification) {
        synchronized(repoLock) {
            val currentList = _notifications.value.toMutableList()
            val index = currentList.indexOfFirst { it.id == item.id }
            if (index != -1) {
                val existing = currentList[index]
                val mergedItem = if (item.mediaUriString.isNullOrBlank() && !existing.mediaUriString.isNullOrBlank()) {
                    item.copy(mediaUriString = existing.mediaUriString)
                } else {
                    item
                }
                currentList[index] = mergedItem
                _notifications.value = currentList
                persistToStorageLocked(currentList)
            } else {
                addNotificationLocked(item)
            }
        }
    }

    fun addNotification(item: InterceptedNotification) {
        synchronized(repoLock) {
            addNotificationLocked(item)
        }
    }

    private fun addNotificationLocked(item: InterceptedNotification) {
        val currentList = _notifications.value.toMutableList()

        val indexById = currentList.indexOfFirst { it.id == item.id }
        if (indexById != -1) {
            val existing = currentList[indexById]
            val mergedItem = if (item.mediaUriString.isNullOrBlank() && !existing.mediaUriString.isNullOrBlank()) {
                item.copy(mediaUriString = existing.mediaUriString)
            } else {
                item
            }
            currentList[indexById] = mergedItem
            _notifications.value = currentList
            persistToStorageLocked(currentList)
            return
        }

        val pendingIndex = currentList.indexOfFirst { existing ->
            existing.packageName == item.packageName &&
                existing.sender.equals(item.sender, ignoreCase = true) &&
                existing.mediaType == item.mediaType &&
                (existing.isPendingDownload() || existing.riskScore == 0.0f) &&
                kotlin.math.abs(item.timestamp - existing.timestamp) < 90000L
        }
        if (pendingIndex != -1) {
            val existing = currentList[pendingIndex]
            val mergedItem = if (item.mediaUriString.isNullOrBlank() && !existing.mediaUriString.isNullOrBlank()) {
                item.copy(mediaUriString = existing.mediaUriString)
            } else {
                item
            }
            currentList[pendingIndex] = mergedItem
            _notifications.value = currentList
            persistToStorageLocked(currentList)
            return
        }

        val existingIndex = currentList.indexOfFirst { existing ->
            existing.packageName == item.packageName &&
                existing.sender.equals(item.sender, ignoreCase = true) &&
                existing.text.equals(item.text, ignoreCase = true) &&
                kotlin.math.abs(item.timestamp - existing.timestamp) < 4000L
        }

        if (existingIndex != -1) {
            val existing = currentList[existingIndex]
            val existingHasMedia = NotificationMediaCache.hasMedia(existing.id)
            val newHasMedia = NotificationMediaCache.hasMedia(item.id)

            if (!existingHasMedia && newHasMedia) {
                currentList[existingIndex] = item
                _notifications.value = currentList
                persistToStorageLocked(currentList)
            }
            return
        }

        val updatedList = (listOf(item) + currentList).take(MAX_SAVED_ITEMS)
        _notifications.value = updatedList
        persistToStorageLocked(updatedList)
    }

    fun autoResolvePendingNotifications(context: Context) {
        val pendingItemsSnapshot = synchronized(repoLock) {
            _notifications.value.filter { it.isPendingDownload() }
        }
        if (pendingItemsSnapshot.isEmpty()) return
        if (!isResolvingPending.compareAndSet(false, true)) return

        val app = context.applicationContext
        pendingResolveExecutor.execute {
            var audioClassifier: AudioClassifier? = null
            var visionClassifier: VisionClassifier? = null
            try {
                val pendingItems = synchronized(repoLock) {
                    _notifications.value.filter { it.isPendingDownload() }
                }
                if (pendingItems.isEmpty()) return@execute

                val needsAudio = pendingItems.any { it.mediaType == MediaType.AUDIO_ONLY || it.mediaType == MediaType.VIDEO_MULTIMODAL }
                val needsVision = pendingItems.any { it.mediaType == MediaType.IMAGE_ONLY || it.mediaType == MediaType.VIDEO_MULTIMODAL }
                if (needsAudio) {
                    audioClassifier = AudioClassifier(app)
                }
                if (needsVision) {
                    visionClassifier = VisionClassifier(app)
                }

                val resolvedMap = HashMap<String, InterceptedNotification>(pendingItems.size)
                for (item in pendingItems) {
                    resolvedMap[item.id] = resolveSinglePendingNotification(app, item, audioClassifier, visionClassifier)
                }

                synchronized(repoLock) {
                    val currentList = _notifications.value.toMutableList()
                    var hasChanges = false
                    for (i in currentList.indices) {
                        val resolved = resolvedMap[currentList[i].id]
                        if (resolved != null && currentList[i].isPendingDownload()) {
                            currentList[i] = resolved
                            hasChanges = true
                        }
                    }
                    if (hasChanges) {
                        _notifications.value = currentList
                        persistToStorageLocked(currentList)
                    }
                }
            } catch (_: Throwable) {
            } finally {
                try { audioClassifier?.close() } catch (_: Throwable) {}
                try { visionClassifier?.close() } catch (_: Throwable) {}
                isResolvingPending.set(false)
            }
        }
    }

    @JvmOverloads
    fun resolveSinglePendingNotification(
        context: Context,
        pending: InterceptedNotification,
        sharedAudioClassifier: AudioClassifier? = null,
        sharedVisionClassifier: VisionClassifier? = null
    ): InterceptedNotification {
        val app = context.applicationContext
        val isBlocked = isSenderBlocked(pending.sender)
        val uriStr = pending.mediaUriString ?: NotificationMediaCache.getMediaUriString(pending.id)
        val cleanOriginal = pending.text
            .replace(Regex("""(?i)\s*\((?:descarga|descargando|pendiente)[^)]*\)"""), "")
            .trim()

        if (pending.mediaType == MediaType.UNKNOWN) {
            val risk = if (isBlocked) maxOf(pending.riskScore, 85.0f) else pending.riskScore
            return pending.copy(
                text = if (isBlocked && !cleanOriginal.startsWith("[REMITENTE BLOQUEADO]")) {
                    "[REMITENTE BLOQUEADO] ${cleanOriginal.ifBlank { "Mensaje analizado" }}"
                } else {
                    cleanOriginal.ifBlank { "Mensaje analizado" }
                },
                riskScore = risk,
                isThreat = isBlocked || risk >= 70.0f
            )
        }

        var localAudio: AudioClassifier? = null
        var localVision: VisionClassifier? = null
        try {
            when (pending.mediaType) {
                MediaType.AUDIO_ONLY -> {
                    val cachedAudio = NotificationMediaCache.getAudioBitmap(app, pending.id)
                    val cachedPcm = NotificationMediaCache.getAudioPcm(pending.id)
                    var specBitmap: Bitmap? = cachedAudio
                    var audioPcm: FloatArray? = cachedPcm
                    if (uriStr != null && (specBitmap == null || audioPcm == null)) {
                        when {
                            uriStr.startsWith("asset://") -> {
                                val assetRef = uriStr.removePrefix("asset://")
                                if (audioPcm == null) {
                                    audioPcm = decodeAssetWavPcm48k(app, wavAssetFromSampleSpec(assetRef))
                                }
                                if (specBitmap == null) {
                                    specBitmap = decodeAssetBitmap(app, specAssetFromAudioAsset(assetRef))
                                        ?: audioPcm?.let { AudioSpectrogramGenerator.analyzePcm(it).spectrogramBitmap }
                                }
                            }
                            uriStr.startsWith("file://") || uriStr.startsWith("content://") -> {
                                val parsed = Uri.parse(uriStr)
                                if (uriStr.endsWith(".png", true) || uriStr.endsWith(".jpg", true)) {
                                    if (specBitmap == null) {
                                        specBitmap = TfliteExecutionLock.decodeSampledBitmapFromUri(app, parsed, 1024)
                                    }
                                } else {
                                    val audioRes = AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(app, parsed)
                                    if (specBitmap == null) specBitmap = audioRes?.spectrogramBitmap
                                    if (audioPcm == null) audioPcm = audioRes?.pcm48k
                                }
                            }
                        }
                    }

                    if (audioPcm == null && specBitmap == null) {
                        return buildUnresolvedLocalFallback(pending, isBlocked)
                    }

                    val audioCls = sharedAudioClassifier ?: AudioClassifier(app).also { localAudio = it }
                    val audioProb = if (audioPcm != null) {
                        audioCls.classifyPcm(audioPcm, null)
                    } else {
                        audioCls.classifySpectrogram(specBitmap!!, null)
                    }
                    val baseFusion = RiskScorer.calculateGlobalRisk(
                        audioProb = audioProb,
                        visionProb = null,
                        isFaceDetected = true
                    )
                    val finalRisk = if (isBlocked) maxOf(baseFusion.globalRiskPercentage, 85.0f) else baseFusion.globalRiskPercentage
                    val finalLevel = if (isBlocked) RiskLevel.ALTO else baseFusion.riskLevel
                    val finalThreat = isBlocked || finalRisk >= 70.0f
                    val finalSummary = if (isBlocked) "[REMITENTE BLOQUEADO] ${baseFusion.diagnosticSummary}" else baseFusion.diagnosticSummary
                    val finalFusion = baseFusion.copy(
                        globalRiskPercentage = finalRisk,
                        riskLevel = finalLevel,
                        diagnosticSummary = finalSummary
                    )

                    val statusText = when {
                        finalRisk >= 70.0f -> "Nota de voz analizada (Alerta: Posible clonación sintética por IA)"
                        finalRisk >= 30.0f -> "Nota de voz analizada (Revisión sugerida: anomalía espectral moderada)"
                        else -> "Nota de voz analizada (Voz humana auténtica)"
                    }
                    val displayText = if (cleanOriginal.isNotBlank() && !cleanOriginal.startsWith("Nota de voz")) {
                        "$cleanOriginal | $statusText"
                    } else {
                        statusText
                    }

                    NotificationMediaCache.storeMedia(
                        context = app,
                        notifId = pending.id,
                        audioBitmap = specBitmap,
                        mediaUriString = uriStr,
                        audioPcm = audioPcm
                    )

                    return pending.copy(
                        text = displayText,
                        riskScore = finalRisk,
                        isThreat = finalThreat,
                        fusionResult = finalFusion,
                        mediaUriString = uriStr
                    )
                }

                MediaType.IMAGE_ONLY -> {
                    val cachedFace = NotificationMediaCache.getFaceBitmap(app, pending.id)
                    val rawBitmap = uriStr?.let { s ->
                        when {
                            s.startsWith("synthetic://") -> createNonFaceLandscapeBitmap()
                            s.startsWith("asset://") -> decodeAssetBitmap(app, s.removePrefix("asset://"))
                            s.startsWith("file://") || s.startsWith("content://") -> {
                                TfliteExecutionLock.decodeSampledBitmapFromUri(app, Uri.parse(s), 1024)
                            }
                            else -> null
                        }
                    } ?: cachedFace

                    if (rawBitmap == null) {
                        return buildUnresolvedLocalFallback(pending, isBlocked)
                    }

                    val visionCls = sharedVisionClassifier ?: VisionClassifier(app).also { localVision = it }
                    val reuseAlreadyCropped = (cachedFace != null && rawBitmap === cachedFace && pending.fusionResult != null && !pending.isFaceDiscarded())
                    val (croppedFace, faceOk) = if (reuseAlreadyCropped) {
                        Pair(rawBitmap, true)
                    } else {
                        visionCls.detectAndCropFace(rawBitmap)
                    }

                    if (!faceOk) {
                        val discardFusion = RiskScorer.calculateGlobalRisk(
                            audioProb = null,
                            visionProb = 0.0f,
                            isFaceDetected = false
                        )
                        val statusText = "Fotografía analizada (BlazeFace: Descarte automático por ausencia de rostro humano)"
                        val displayText = if (cleanOriginal.isNotBlank() && !cleanOriginal.startsWith("Fotografía")) {
                            "$cleanOriginal | $statusText"
                        } else {
                            statusText
                        }

                        NotificationMediaCache.storeMedia(
                            context = app,
                            notifId = pending.id,
                            faceBitmap = croppedFace,
                            mediaUriString = uriStr
                        )

                        return pending.copy(
                            text = displayText,
                            riskScore = 0.0f,
                            isThreat = false,
                            fusionResult = discardFusion,
                            mediaUriString = uriStr
                        )
                    }

                    val visionProb = visionCls.classifyFaceKeyframe(croppedFace, null, requireFaceDetection = false)
                    val visionSharp = VisionClassifier.computeLaplacianSharpness(croppedFace)
                    val baseFusion = RiskScorer.calculateGlobalRisk(
                        audioProb = null,
                        visionProb = visionProb,
                        isFaceDetected = true,
                        visionSharpness = visionSharp
                    )
                    val finalRisk = if (isBlocked) maxOf(baseFusion.globalRiskPercentage, 85.0f) else baseFusion.globalRiskPercentage
                    val finalLevel = if (isBlocked) RiskLevel.ALTO else baseFusion.riskLevel
                    val finalThreat = isBlocked || finalRisk >= 70.0f
                    val finalSummary = if (isBlocked) "[REMITENTE BLOQUEADO] ${baseFusion.diagnosticSummary}" else baseFusion.diagnosticSummary
                    val finalFusion = baseFusion.copy(
                        globalRiskPercentage = finalRisk,
                        riskLevel = finalLevel,
                        diagnosticSummary = finalSummary
                    )

                    val statusText = when {
                        finalRisk >= 70.0f -> "Fotografía analizada (Alerta: Posible manipulación facial por IA)"
                        finalRisk >= 30.0f -> "Fotografía analizada (Revisión sugerida: anomalía facial moderada)"
                        else -> "Fotografía analizada (Rostro humano auténtico)"
                    }
                    val displayText = if (cleanOriginal.isNotBlank() && !cleanOriginal.startsWith("Fotografía")) {
                        "$cleanOriginal | $statusText"
                    } else {
                        statusText
                    }

                    NotificationMediaCache.storeMedia(
                        context = app,
                        notifId = pending.id,
                        faceBitmap = croppedFace,
                        mediaUriString = uriStr
                    )

                    return pending.copy(
                        text = displayText,
                        riskScore = finalRisk,
                        isThreat = finalThreat,
                        fusionResult = finalFusion,
                        mediaUriString = uriStr
                    )
                }

                MediaType.VIDEO_MULTIMODAL -> {
                    val cachedFace = NotificationMediaCache.getFaceBitmap(app, pending.id)
                    val cachedAudio = NotificationMediaCache.getAudioBitmap(app, pending.id)
                    val cachedPcm = NotificationMediaCache.getAudioPcm(pending.id)
                    val visionCls = sharedVisionClassifier ?: VisionClassifier(app).also { localVision = it }
                    val audioCls = sharedAudioClassifier ?: AudioClassifier(app).also { localAudio = it }

                    val isRealVideoUri = uriStr != null &&
                        (uriStr.startsWith("file://") || uriStr.startsWith("content://")) &&
                        !uriStr.endsWith(".jpg", true) &&
                        !uriStr.endsWith(".png", true)

                    val frameCrops = mutableListOf<Bitmap>()
                    val frameQualities = mutableListOf<Float>()
                    var faceOk = false
                    var audioSpec: Bitmap? = cachedAudio
                    var audioPcm: FloatArray? = cachedPcm
                    var audioConf = 1.0f

                    if (isRealVideoUri) {
                        val parsed = Uri.parse(uriStr)
                        val extRes = visionCls.extractBestFiveVideoKeyframes(parsed, forceTightCrop = true, maxFrames = 5)
                        frameCrops.addAll(extRes.crops)
                        frameQualities.addAll(extRes.qualities)
                        faceOk = extRes.hasFace
                        if (audioSpec == null || audioPcm == null) {
                            val audioRes = AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(app, parsed)
                            if (audioSpec == null) audioSpec = audioRes?.spectrogramBitmap
                            if (audioPcm == null) audioPcm = audioRes?.pcmFull16k ?: audioRes?.pcm48k
                            if (audioRes != null) audioConf = audioRes.audioConfidence
                        }
                    } else if (uriStr != null && uriStr.startsWith("asset://")) {
                        val assetBmp = decodeAssetBitmap(app, uriStr.removePrefix("asset://"))
                        if (assetBmp != null) {
                            val (crop, ok) = visionCls.detectAndCropFace(assetBmp)
                            faceOk = ok
                            frameCrops.add(crop)
                            if (ok) frameQualities.add(VisionClassifier.computeLaplacianSharpness(crop))
                        }
                        if (audioSpec == null) {
                            val pairedSpec = if (uriStr.contains("06_deepfake")) {
                                "samples/06_clonacion_ia_spoof_tts_1_spec.png"
                            } else {
                                "samples/01_voz_humana_real_bonafide_1_spec.png"
                            }
                            audioSpec = decodeAssetBitmap(app, pairedSpec)
                        }
                        if (audioPcm == null) {
                            val pairedWav = if (uriStr.contains("06_deepfake")) {
                                "samples/06_clonacion_ia_spoof_tts_1.wav"
                            } else {
                                "samples/01_voz_humana_real_bonafide_1.wav"
                            }
                            audioPcm = decodeAssetWavPcm48k(app, pairedWav)
                        }
                    } else if (cachedFace != null || cachedAudio != null || cachedPcm != null) {
                        if (cachedFace != null) {
                            faceOk = !pending.isFaceDiscarded()
                            frameCrops.add(cachedFace)
                            if (faceOk) frameQualities.add(VisionClassifier.computeLaplacianSharpness(cachedFace))
                        }
                    } else {
                        return buildUnresolvedLocalFallback(pending, isBlocked)
                    }

                    if (frameCrops.isEmpty() && audioPcm == null && audioSpec == null) {
                        return buildUnresolvedLocalFallback(pending, isBlocked)
                    }

                    val frameProbs = if (faceOk && frameCrops.isNotEmpty()) {
                        frameCrops.map { frame ->
                            visionCls.classifyFaceKeyframe(frame, null, requireFaceDetection = false)
                        }
                    } else {
                        emptyList()
                    }
                    val audioProb = when {
                        audioPcm != null -> {
                            audioConf = AudioSpectrogramGenerator.estimateAudioConfidence(audioPcm)
                            audioCls.classifyPcm(audioPcm, null)
                        }
                        audioSpec != null -> audioCls.classifySpectrogram(audioSpec, null)
                        else -> null
                    }

                    val baseFusion = RiskScorer.calculateVideoRisk(
                        audioProb = audioProb,
                        frameProbs = frameProbs,
                        isFaceDetected = faceOk,
                        frameQualities = frameQualities,
                        audioConfidence = audioConf
                    )
                    val finalRisk = if (isBlocked) maxOf(baseFusion.globalRiskPercentage, 85.0f) else baseFusion.globalRiskPercentage
                    val finalLevel = if (isBlocked) RiskLevel.ALTO else baseFusion.riskLevel
                    val finalThreat = isBlocked || finalRisk >= 70.0f
                    val finalSummary = if (isBlocked) "[REMITENTE BLOQUEADO] ${baseFusion.diagnosticSummary}" else baseFusion.diagnosticSummary
                    val finalFusion = baseFusion.copy(
                        globalRiskPercentage = finalRisk,
                        riskLevel = finalLevel,
                        diagnosticSummary = finalSummary
                    )

                    val statusText = when {
                        finalRisk >= 70.0f -> "Video analizado (Alerta: Inconsistencia multimodal detectada)"
                        finalRisk >= 30.0f -> "Video analizado (Revisión sugerida: anomalía multimodal moderada)"
                        else -> "Video analizado (Contenido audiovisual auténtico)"
                    }
                    val displayText = if (cleanOriginal.isNotBlank() && !cleanOriginal.startsWith("Video")) {
                        "$cleanOriginal | $statusText"
                    } else {
                        statusText
                    }

                    NotificationMediaCache.storeMedia(
                        context = app,
                        notifId = pending.id,
                        faceBitmap = frameCrops.firstOrNull(),
                        audioBitmap = audioSpec,
                        mediaUriString = uriStr,
                        audioPcm = audioPcm
                    )

                    return pending.copy(
                        text = displayText,
                        riskScore = finalRisk,
                        isThreat = finalThreat,
                        fusionResult = finalFusion,
                        mediaUriString = uriStr
                    )
                }

                MediaType.UNKNOWN -> return pending
            }
        } finally {
            try { localAudio?.close() } catch (_: Throwable) {}
            try { localVision?.close() } catch (_: Throwable) {}
        }
    }

    private fun buildUnresolvedLocalFallback(
        pending: InterceptedNotification,
        isBlocked: Boolean
    ): InterceptedNotification {
        val risk = if (isBlocked) 85.0f else 0.0f
        val fusion = FusionResult(
            globalRiskPercentage = risk,
            audioFraudProb = 0.0f,
            visionFraudProb = 0.0f,
            riskLevel = if (isBlocked) RiskLevel.ALTO else RiskLevel.BAJO,
            weightAudio = if (pending.mediaType == MediaType.AUDIO_ONLY) 1.0f else if (pending.mediaType == MediaType.VIDEO_MULTIMODAL) 0.55f else 0.0f,
            weightVision = if (pending.mediaType == MediaType.IMAGE_ONLY) 1.0f else if (pending.mediaType == MediaType.VIDEO_MULTIMODAL) 0.45f else 0.0f,
            diagnosticSummary = "Archivo multimedia no localizado en almacenamiento local (habilita Acceso Total o descarga manual para analizar)."
        )
        val baseText = "Archivo pendiente sin acceso local"
        return pending.copy(
            text = if (isBlocked) "[REMITENTE BLOQUEADO] $baseText" else baseText,
            riskScore = risk,
            isThreat = isBlocked,
            fusionResult = fusion
        )
    }

    private fun decodeAssetBitmap(context: Context, assetPath: String): Bitmap? {
        return try {
            context.assets.open(assetPath).use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (_: Throwable) {
            null
        }
    }

    private fun decodeAssetWavPcm48k(context: Context, assetPath: String): FloatArray? {
        return try {
            context.assets.open(assetPath).use { stream ->
                AudioSpectrogramGenerator.buildPcm48kFromWavBytes(stream.readBytes())
            }
        } catch (_: Throwable) {
            null
        }
    }

    private fun wavAssetFromSampleSpec(assetPath: String): String {
        return when {
            assetPath.endsWith(".wav", ignoreCase = true) -> assetPath
            assetPath == "samples/audio_real_spec.png" -> "samples/01_voz_humana_real_bonafide_1.wav"
            assetPath == "samples/audio_fake_spec.png" -> "samples/06_clonacion_ia_spoof_tts_1.wav"
            assetPath.endsWith("_spec.png") -> assetPath.removeSuffix("_spec.png") + ".wav"
            else -> "samples/01_voz_humana_real_bonafide_1.wav"
        }
    }

    private fun specAssetFromAudioAsset(assetPath: String): String {
        return if (assetPath.endsWith(".wav", ignoreCase = true)) {
            assetPath.removeSuffix(".wav") + "_spec.png"
        } else {
            assetPath
        }
    }

    private fun createNonFaceLandscapeBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()
        canvas.drawColor(Color.rgb(12, 74, 110))
        paint.color = Color.rgb(3, 105, 161)
        canvas.drawRect(0f, 70f, 224f, 140f, paint)
        val mountainPath = Path().apply {
            moveTo(0f, 175f)
            lineTo(55f, 95f)
            lineTo(118f, 155f)
            lineTo(175f, 88f)
            lineTo(224f, 175f)
            close()
        }
        paint.color = Color.rgb(30, 41, 59)
        canvas.drawPath(mountainPath, paint)
        paint.color = Color.rgb(20, 83, 45)
        canvas.drawRect(0f, 175f, 224f, 224f, paint)
        return bitmap
    }

    private fun extractVideoFaceKeyframes(
        context: Context,
        uri: Uri,
        visionClassifier: VisionClassifier,
        maxFrames: Int = 5
    ): Pair<List<Bitmap>, Boolean> {
        val extRes = visionClassifier.extractBestFiveVideoKeyframes(
            uri = uri,
            forceTightCrop = true,
            maxFrames = maxFrames
        )
        return Pair(extRes.crops, extRes.hasFace)
    }

    fun clearNotifications() {
        synchronized(repoLock) {
            _notifications.value = emptyList()
            prefs?.edit()?.remove(KEY_NOTIFS)?.apply()
        }
    }

    fun isPermissionGranted(context: Context): Boolean {
        val pkgName = context.packageName

        try {
            val enabledListeners = androidx.core.app.NotificationManagerCompat.getEnabledListenerPackages(context)
            if (enabledListeners.contains(pkgName)) {
                return true
            }
        } catch (e: Exception) {
        }

        try {
            val flat = Settings.Secure.getString(context.contentResolver, "enabled_notification_listeners")
            if (!flat.isNullOrEmpty()) {
                val names = flat.split(":")
                for (name in names) {
                    val cn = ComponentName.unflattenFromString(name)
                    if (cn != null && cn.packageName == pkgName) {
                        return true
                    }
                }
            }
        } catch (e: Exception) {
        }

        return false
    }
}

object NotificationMediaCache {
    private val faceCache = ConcurrentHashMap<String, Bitmap>()
    private val audioCache = ConcurrentHashMap<String, Bitmap>()
    private val pcmCache = ConcurrentHashMap<String, FloatArray>()
    private val assetNameCache = ConcurrentHashMap<String, String>()
    private val mediaUriCache = ConcurrentHashMap<String, String>()
    private val boundFiles = Collections.synchronizedSet(HashSet<String>())

    fun isFileBound(pathOrUri: String): Boolean = boundFiles.contains(pathOrUri)

    fun markFileBound(pathOrUri: String) {
        boundFiles.add(pathOrUri)
    }

    fun releaseFile(pathOrUri: String) {
        boundFiles.remove(pathOrUri)
    }

    fun storeMedia(
        context: Context? = null,
        notifId: String,
        faceBitmap: Bitmap? = null,
        audioBitmap: Bitmap? = null,
        assetName: String? = null,
        mediaUriString: String? = null,
        audioPcm: FloatArray? = null
    ) {
        if (faceCache.size > 30) {
            try {
                val key = faceCache.keys().nextElement()
                faceCache.remove(key)
            } catch (e: Exception) {}
        }
        if (audioCache.size > 30) {
            try {
                val key = audioCache.keys().nextElement()
                audioCache.remove(key)
            } catch (e: Exception) {}
        }
        if (pcmCache.size > 30) {
            try {
                val key = pcmCache.keys().nextElement()
                pcmCache.remove(key)
            } catch (e: Exception) {}
        }

        faceBitmap?.takeIf { !it.isRecycled && it.width > 0 && it.height > 0 }?.let {
            faceCache[notifId] = it
            if (context != null) saveBitmapToCache(context, "face_$notifId.png", it)
        }
        audioBitmap?.takeIf { !it.isRecycled && it.width > 0 && it.height > 0 }?.let {
            audioCache[notifId] = it
            if (context != null) saveBitmapToCache(context, "audio_$notifId.png", it)
        }
        audioPcm?.takeIf { it.isNotEmpty() }?.let {
            pcmCache[notifId] = it
        }
        assetName?.let { assetNameCache[notifId] = it }
        mediaUriString?.let {
            mediaUriCache[notifId] = it
            boundFiles.add(it)
        }
    }

    fun hasMedia(notifId: String): Boolean =
        faceCache.containsKey(notifId) ||
            audioCache.containsKey(notifId) ||
            pcmCache.containsKey(notifId) ||
            mediaUriCache.containsKey(notifId)

    fun getAudioPcm(notifId: String): FloatArray? = pcmCache[notifId]

    fun getMediaUriString(notifId: String): String? = mediaUriCache[notifId]

    fun getFaceBitmap(context: Context? = null, notifId: String): Bitmap? {
        faceCache[notifId]?.let { cached ->
            if (!cached.isRecycled && cached.width > 0 && cached.height > 0) {
                return cached
            }
            faceCache.remove(notifId)
        }
        if (context != null) {
            loadBitmapFromCache(context, "face_$notifId.png")?.let {
                faceCache[notifId] = it
                return it
            }
        }
        return null
    }

    fun getFaceBitmap(notifId: String): Bitmap? = getFaceBitmap(null, notifId)

    fun getAudioBitmap(context: Context? = null, notifId: String): Bitmap? {
        audioCache[notifId]?.let { cached ->
            if (!cached.isRecycled && cached.width > 0 && cached.height > 0) {
                return cached
            }
            audioCache.remove(notifId)
        }
        if (context != null) {
            loadBitmapFromCache(context, "audio_$notifId.png")?.let {
                audioCache[notifId] = it
                return it
            }
        }
        return null
    }

    fun getAudioBitmap(notifId: String): Bitmap? = getAudioBitmap(null, notifId)

    fun getAssetName(notifId: String): String? = assetNameCache[notifId]

    private fun saveBitmapToCache(context: Context, filename: String, bitmap: Bitmap) {
        if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) return
        try {
            val file = java.io.File(context.cacheDir, filename)
            java.io.FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
        } catch (_: Exception) {
        }
    }

    private fun loadBitmapFromCache(context: Context, filename: String): Bitmap? {
        return try {
            val file = java.io.File(context.cacheDir, filename)
            if (file.isFile && file.length() > 0) {
                com.detectorpreventor.app.ml.TfliteExecutionLock.decodeSampledBitmapFromFile(file.absolutePath, 1024)
            } else {
                null
            }
        } catch (_: Exception) {
            null
        }
    }

    fun clear() {
        faceCache.clear()
        audioCache.clear()
        pcmCache.clear()
        assetNameCache.clear()
        mediaUriCache.clear()
        boundFiles.clear()
    }
}
