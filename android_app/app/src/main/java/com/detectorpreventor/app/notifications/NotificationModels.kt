package com.detectorpreventor.app.notifications

import android.content.ComponentName
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.provider.Settings
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

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
    val fusionResult: FusionResult? = null
) {
    val formattedTime: String
        get() = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(timestamp))

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
        fusionResult?.let { fr ->
            val frObj = JSONObject()
            frObj.put("globalRiskPercentage", fr.globalRiskPercentage.toDouble())
            frObj.put("audioFraudProb", fr.audioFraudProb.toDouble())
            frObj.put("visionFraudProb", fr.visionFraudProb.toDouble())
            frObj.put("riskLevel", fr.riskLevel.name)
            frObj.put("weightAudio", fr.weightAudio.toDouble())
            frObj.put("weightVision", fr.weightVision.toDouble())
            frObj.put("diagnosticSummary", fr.diagnosticSummary)
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
                    diagnosticSummary = frObj.optString("diagnosticSummary", "")
                )
            } else null

            val mType = try {
                MediaType.valueOf(obj.optString("mediaType", MediaType.UNKNOWN.name))
            } catch (e: Exception) {
                MediaType.UNKNOWN
            }

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
                fusionResult = fr
            )
        }
    }
}

object NotificationRepository {
    private const val PREFS_NAME = "detector_notifications_store"
    private const val KEY_NOTIFS = "saved_notifications"
    private const val MAX_SAVED_ITEMS = 50
    private const val DEDUPLICATION_WINDOW_MS = 8000L

    private var prefs: SharedPreferences? = null

    private val _notifications = MutableStateFlow<List<InterceptedNotification>>(emptyList())
    val notifications: StateFlow<List<InterceptedNotification>> = _notifications.asStateFlow()

    private val _isMonitoringActive = MutableStateFlow(true)
    val isMonitoringActive: StateFlow<Boolean> = _isMonitoringActive.asStateFlow()

    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    fun init(context: Context) {
        val app = context.applicationContext
        prefs = app.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        loadFromStorage()
    }

    private fun loadFromStorage() {
        val jsonString = prefs?.getString(KEY_NOTIFS, null) ?: return
        try {
            val jsonArray = JSONArray(jsonString)
            val list = mutableListOf<InterceptedNotification>()
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                list.add(InterceptedNotification.fromJson(obj))
            }
            _notifications.value = list
        } catch (e: Exception) {
        }
    }

    private fun persistToStorage(list: List<InterceptedNotification>) {
        val p = prefs ?: return
        try {
            val jsonArray = JSONArray()
            list.forEach { jsonArray.put(it.toJson()) }
            p.edit().putString(KEY_NOTIFS, jsonArray.toString()).apply()
        } catch (e: Exception) {
        }
    }

    fun setMonitoringActive(active: Boolean) {
        _isMonitoringActive.value = active
    }

    fun setServiceConnected(connected: Boolean) {
        _isServiceConnected.value = connected
    }

    fun addNotification(item: InterceptedNotification) {
        val currentList = _notifications.value.toMutableList()

        val existingIndex = currentList.indexOfFirst { existing ->
            existing.packageName == item.packageName &&
                    existing.sender.equals(item.sender, ignoreCase = true) &&
                    kotlin.math.abs(item.timestamp - existing.timestamp) < DEDUPLICATION_WINDOW_MS
        }

        if (existingIndex != -1) {
            val existing = currentList[existingIndex]
            val existingHasMedia = NotificationMediaCache.hasMedia(existing.id)
            val newHasMedia = NotificationMediaCache.hasMedia(item.id)

            if (!existingHasMedia && newHasMedia) {
                currentList[existingIndex] = item
                _notifications.value = currentList
                persistToStorage(currentList)
                return
            } else if (existingHasMedia && !newHasMedia) {
                return
            } else {
                currentList[existingIndex] = item
                _notifications.value = currentList
                persistToStorage(currentList)
                return
            }
        }

        val updatedList = (listOf(item) + currentList).take(MAX_SAVED_ITEMS)
        _notifications.value = updatedList
        persistToStorage(updatedList)
    }

    fun clearNotifications() {
        _notifications.value = emptyList()
        prefs?.edit()?.remove(KEY_NOTIFS)?.apply()
    }

    fun isPermissionGranted(context: Context): Boolean {
        val pkgName = context.packageName
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
        return false
    }
}

object NotificationMediaCache {
    private val faceCache = ConcurrentHashMap<String, Bitmap>()
    private val audioCache = ConcurrentHashMap<String, Bitmap>()
    private val assetNameCache = ConcurrentHashMap<String, String>()

    fun storeMedia(
        context: Context? = null,
        notifId: String,
        faceBitmap: Bitmap? = null,
        audioBitmap: Bitmap? = null,
        assetName: String? = null
    ) {
        faceBitmap?.let {
            faceCache[notifId] = it
            if (context != null) saveBitmapToCache(context, "face_$notifId.png", it)
        }
        audioBitmap?.let {
            audioCache[notifId] = it
            if (context != null) saveBitmapToCache(context, "audio_$notifId.png", it)
        }
        assetName?.let { assetNameCache[notifId] = it }
    }

    fun hasMedia(notifId: String): Boolean =
        faceCache.containsKey(notifId) || audioCache.containsKey(notifId)

    fun getFaceBitmap(context: Context? = null, notifId: String): Bitmap? {
        faceCache[notifId]?.let { return it }
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
        audioCache[notifId]?.let { return it }
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
        try {
            val file = java.io.File(context.cacheDir, filename)
            java.io.FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
        } catch (e: Exception) {
        }
    }

    private fun loadBitmapFromCache(context: Context, filename: String): Bitmap? {
        return try {
            val file = java.io.File(context.cacheDir, filename)
            if (file.exists() && file.length() > 0) {
                android.graphics.BitmapFactory.decodeFile(file.absolutePath)
            } else null
        } catch (e: Exception) {
            null
        }
    }

    fun clear() {
        faceCache.clear()
        audioCache.clear()
        assetNameCache.clear()
    }
}


