package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import java.io.InputStream

object TfliteExecutionLock {
    private const val TAG = "TfliteExecutionLock"

    val nativeLock = Any()
    val faceLock = Any()

    fun decodeSampledBitmapFromFile(path: String, maxDim: Int = 1024): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(path, bounds)
            val sampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight, maxDim)
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val decoded = BitmapFactory.decodeFile(path, decodeOptions) ?: return null
            capBitmapDimension(decoded, maxDim)
        } catch (t: Throwable) {
            Log.w(TAG, "Error decodificando archivo bitmap ($path): ${t.message}")
            null
        }
    }

    fun decodeSampledBitmapFromUri(context: Context, uri: Uri, maxDim: Int = 1024): Bitmap? {
        return decodeSampledBitmapFromStream(
            openStream = { context.contentResolver.openInputStream(uri) },
            maxDim = maxDim
        )
    }

    fun decodeSampledBitmapFromStream(
        openStream: () -> InputStream?,
        maxDim: Int = 1024
    ): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            openStream()?.use { stream ->
                BitmapFactory.decodeStream(stream, null, bounds)
            }
            val sampleSize = if (bounds.outWidth > 0 && bounds.outHeight > 0) {
                calculateInSampleSize(bounds.outWidth, bounds.outHeight, maxDim)
            } else {
                1
            }
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val decoded = openStream()?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOptions)
            } ?: return null
            capBitmapDimension(decoded, maxDim)
        } catch (t: Throwable) {
            Log.w(TAG, "Error decodificando stream bitmap: ${t.message}")
            null
        }
    }

    fun calculateInSampleSize(width: Int, height: Int, maxDim: Int): Int {
        if (width <= 0 || height <= 0 || maxDim <= 0) return 1
        var sampleSize = 1
        val largest = maxOf(width, height)
        while ((largest / sampleSize) > maxDim) {
            sampleSize *= 2
        }
        return sampleSize
    }

    private fun capBitmapDimension(bitmap: Bitmap, maxDim: Int): Bitmap {
        val currentMax = maxOf(bitmap.width, bitmap.height)
        if (currentMax <= maxDim || bitmap.width <= 0 || bitmap.height <= 0) {
            return bitmap
        }
        val scale = maxDim.toFloat() / currentMax.toFloat()
        val targetW = maxOf(1, (bitmap.width * scale).toInt())
        val targetH = maxOf(1, (bitmap.height * scale).toInt())
        val scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        if (scaled !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }
        return scaled
    }
}

