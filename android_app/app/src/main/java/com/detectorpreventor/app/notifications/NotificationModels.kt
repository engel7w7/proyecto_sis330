package com.detectorpreventor.app.notifications

import android.content.ComponentName
import android.content.Context
import android.content.SharedPreferences
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
    private const val DEDUPLICATION_WINDOW_MS = 2000L

    private var appContext: Context? = null
    private var prefs: SharedPreferences? = null

    private val _notifications = MutableStateFlow<List<InterceptedNotification>>(emptyList())
    val notifications: StateFlow<List<InterceptedNotification>> = _notifications.asStateFlow()

    private val _isMonitoringActive = MutableStateFlow(true)
    val isMonitoringActive: StateFlow<Boolean> = _isMonitoringActive.asStateFlow()

    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    fun init(context: Context) {
        val app = context.applicationContext
        appContext = app
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
            // Ignorar corrupción y continuar
        }
    }

    private fun persistToStorage(list: List<InterceptedNotification>) {
        val p = prefs ?: return
        try {
            val jsonArray = JSONArray()
            list.forEach { jsonArray.put(it.toJson()) }
            p.edit().putString(KEY_NOTIFS, jsonArray.toString()).apply()
        } catch (e: Exception) {
            // Manejo de excepción silenciosa
        }
    }

    fun setMonitoringActive(active: Boolean) {
        _isMonitoringActive.value = active
    }

    fun setServiceConnected(connected: Boolean) {
        _isServiceConnected.value = connected
    }

    fun addNotification(item: InterceptedNotification) {
        val currentList = _notifications.value

        val isDuplicate = currentList.firstOrNull()?.let { last ->
            last.packageName == item.packageName &&
                    last.sender == item.sender &&
                    last.text == item.text &&
                    (item.timestamp - last.timestamp) < DEDUPLICATION_WINDOW_MS
        } ?: false

        if (isDuplicate) return

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
