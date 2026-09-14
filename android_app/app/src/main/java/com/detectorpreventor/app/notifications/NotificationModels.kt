package com.detectorpreventor.app.notifications

import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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
}

object NotificationRepository {
    private val _notifications = MutableStateFlow<List<InterceptedNotification>>(emptyList())
    val notifications: StateFlow<List<InterceptedNotification>> = _notifications.asStateFlow()

    private val _isMonitoringActive = MutableStateFlow(true)
    val isMonitoringActive: StateFlow<Boolean> = _isMonitoringActive.asStateFlow()

    fun setMonitoringActive(active: Boolean) {
        _isMonitoringActive.value = active
    }

    fun addNotification(item: InterceptedNotification) {
        _notifications.value = listOf(item) + _notifications.value
    }

    fun clearNotifications() {
        _notifications.value = emptyList()
    }
}
