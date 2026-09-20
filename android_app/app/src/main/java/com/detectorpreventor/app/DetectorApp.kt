package com.detectorpreventor.app

import android.app.Application
import com.detectorpreventor.app.notifications.NotificationMonitorService
import com.detectorpreventor.app.notifications.NotificationRepository

/**
 * Punto de entrada principal de la aplicacion Android Detector Preventor.
 */
class DetectorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        NotificationRepository.init(this)
        NotificationMonitorService.ensureServiceBound(this)
    }
}


