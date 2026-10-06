package com.detectorpreventor.app

import android.app.Application
import com.detectorpreventor.app.notifications.NotificationMonitorService
import com.detectorpreventor.app.notifications.NotificationRepository

class DetectorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        NotificationRepository.init(this)
        NotificationMonitorService.ensureServiceBound(this)
    }
}


