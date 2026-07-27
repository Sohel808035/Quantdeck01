package com.quantspherex.app

import android.app.Application
import android.util.Log

class QuantSphereXApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i("QuantSphereXApp", "QuantSphereX Android Application v2.0.0 Initialized.")
    }
}
