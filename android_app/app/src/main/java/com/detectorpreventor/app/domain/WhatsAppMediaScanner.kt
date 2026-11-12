package com.detectorpreventor.app.domain

import android.Manifest
import android.content.ContentUris
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.MediaStore
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class WhatsAppAudioItem(
    val uri: Uri? = null,
    val displayName: String,
    val categoryLabel: String,
    val folderHint: String,
    val sizeBytes: Long,
    val lastModifiedMs: Long,
    val sourceKind: String
) {
    val formattedDate: String
        get() = SimpleDateFormat("dd/MM HH:mm", Locale.getDefault()).format(Date(lastModifiedMs))

    val formattedSize: String
        get() = when {
            sizeBytes >= 1_048_576L -> String.format(Locale.US, "%.1f MB", sizeBytes / 1_048_576.0)
            sizeBytes >= 1024L -> String.format(Locale.US, "%.0f KB", sizeBytes / 1024.0)
            else -> "$sizeBytes B"
        }
}

data class WhatsAppStorageAccessState(
    val hasAllFilesAccess: Boolean,
    val hasAudioMediaPermission: Boolean,
    val persistedTreeCount: Int,
    val hasAccessibleWhatsAppDirs: Boolean
)

object WhatsAppMediaScanner {

    private val SUPPORTED_AUDIO_EXTENSIONS = setOf(
        "opus", "m4a", "mp3", "ogg", "aac", "amr", "wav", "flac", "3ga"
    )

    private const val MIN_AUDIO_BYTES = 1024L

    fun isSupportedWhatsAppAudioExtension(fileName: String): Boolean {
        val ext = fileName.substringAfterLast('.', "").lowercase(Locale.ROOT)
        return ext.isNotEmpty() && ext in SUPPORTED_AUDIO_EXTENSIONS
    }

    fun shouldSkipFileName(name: String): Boolean {
        if (name.isBlank()) return true
        val lower = name.lowercase(Locale.ROOT)
        return lower.startsWith(".") ||
            lower == ".nomedia" ||
            lower.endsWith(".tmp") ||
            lower.endsWith(".pending") ||
            lower.endsWith(".enc") ||
            lower.endsWith(".download") ||
            lower.endsWith(".crdownload") ||
            lower.endsWith(".part") ||
            !isSupportedWhatsAppAudioExtension(lower)
    }

    fun inferCategoryLabel(pathOrName: String): String {
        val lower = pathOrName.lowercase(Locale.ROOT)
        return when {
            lower.contains("voice notes") || lower.contains("ptt-") -> "Nota de Voz"
            lower.contains("whatsapp audio") || lower.contains("aud-") -> "Audio WhatsApp"
            lower.contains("whatsapp business") -> "WhatsApp Business"
            else -> "Audio Local"
        }
    }

    fun sortWhatsAppAudioItems(items: List<WhatsAppAudioItem>): List<WhatsAppAudioItem> {
        return items.sortedWith(
            compareByDescending<WhatsAppAudioItem> { it.lastModifiedMs }
                .thenBy { it.displayName }
        )
    }

    fun getStorageAccessState(context: Context): WhatsAppStorageAccessState {
        val hasAllFiles = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            true
        }

        val hasAudioPerm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_AUDIO) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }

        val trees = try {
            context.contentResolver.persistedUriPermissions.count { it.isReadPermission }
        } catch (_: Exception) {
            0
        }

        val accessibleDirExists = candidateWhatsAppRoots().any { it.exists() && it.canRead() }
        return WhatsAppStorageAccessState(
            hasAllFilesAccess = hasAllFiles,
            hasAudioMediaPermission = hasAudioPerm,
            persistedTreeCount = trees,
            hasAccessibleWhatsAppDirs = accessibleDirExists
        )
    }

    fun candidateWhatsAppRoots(): List<File> {
        val storageRoots = linkedSetOf<File>()
        try {
            storageRoots.add(Environment.getExternalStorageDirectory())
        } catch (_: Exception) {
        }
        storageRoots.add(File("/storage/emulated/0"))

        val relativePaths = listOf(
            "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes",
            "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio",
            "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/Sent",
            "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/Private",
            "Android/media/com.whatsapp.w4b/WhatsApp Business/Media/WhatsApp Voice Notes",
            "Android/media/com.whatsapp.w4b/WhatsApp Business/Media/WhatsApp Audio",
            "Android/media/com.whatsapp.w4b/WhatsApp Business/Media/WhatsApp Audio/Sent",
            "Android/media/com.whatsapp.w4b/WhatsApp Business/Media/WhatsApp Audio/Private",
            "WhatsApp/Media/WhatsApp Voice Notes",
            "WhatsApp/Media/WhatsApp Audio",
            "WhatsApp/Media/WhatsApp Audio/Sent",
            "WhatsApp/Media/WhatsApp Audio/Private",
            "WhatsApp Business/Media/WhatsApp Voice Notes",
            "WhatsApp Business/Media/WhatsApp Audio",
            "WhatsApp Business/Media/WhatsApp Audio/Sent",
            "WhatsApp Business/Media/WhatsApp Audio/Private"
        )

        val dirs = mutableListOf<File>()
        for (root in storageRoots) {
            for (rel in relativePaths) {
                dirs.add(File(root, rel))
            }
        }
        return dirs.distinctBy { it.absolutePath }
    }

    fun scanWhatsAppAudio(context: Context, maxItems: Int = 60): List<WhatsAppAudioItem> {
        val deduped = LinkedHashMap<String, WhatsAppAudioItem>()

        for (rootDir in candidateWhatsAppRoots()) {
            try {
                if (!rootDir.exists() || !rootDir.isDirectory || !rootDir.canRead()) continue
                rootDir.walkTopDown()
                    .maxDepth(4)
                    .forEach { file ->
                        if (file.isFile &&
                            !shouldSkipFileName(file.name) &&
                            file.length() >= MIN_AUDIO_BYTES
                        ) {
                            val key = "${file.name}|${file.length()}"
                            if (!deduped.containsKey(key)) {
                                val parentName = file.parentFile?.name ?: rootDir.name
                                deduped[key] = WhatsAppAudioItem(
                                    uri = Uri.fromFile(file),
                                    displayName = file.name,
                                    categoryLabel = inferCategoryLabel("${rootDir.absolutePath}/${file.name}"),
                                    folderHint = parentName,
                                    sizeBytes = file.length(),
                                    lastModifiedMs = file.lastModified(),
                                    sourceKind = "Archivo Directo"
                                )
                            }
                        }
                    }
            } catch (_: Throwable) {
            }
        }

        try {
            val perms = context.contentResolver.persistedUriPermissions
            for (perm in perms) {
                if (!perm.isReadPermission) continue
                val treeUri = perm.uri
                val rootDocId = DocumentsContract.getTreeDocumentId(treeUri)
                scanDocumentTreeRecursive(
                    context = context,
                    treeUri = treeUri,
                    parentDocId = rootDocId,
                    folderLabel = rootDocId.substringAfterLast('/'),
                    depth = 0,
                    maxDepth = 6,
                    outMap = deduped,
                    maxCollect = maxItems * 2
                )
            }
        } catch (_: Throwable) {
        }

        try {
            val projection = mutableListOf(
                MediaStore.Audio.Media._ID,
                MediaStore.Audio.Media.DISPLAY_NAME,
                MediaStore.Audio.Media.DATE_MODIFIED,
                MediaStore.Audio.Media.SIZE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                projection.add(MediaStore.MediaColumns.RELATIVE_PATH)
                projection.add(MediaStore.MediaColumns.IS_PENDING)
            }
            val selection = "${MediaStore.Audio.Media.SIZE} >= $MIN_AUDIO_BYTES"
            val sortOrder = "${MediaStore.Audio.Media.DATE_MODIFIED} DESC"
            context.contentResolver.query(
                MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                projection.toTypedArray(),
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                val idCol = cursor.getColumnIndex(MediaStore.Audio.Media._ID)
                val nameCol = cursor.getColumnIndex(MediaStore.Audio.Media.DISPLAY_NAME)
                val dateCol = cursor.getColumnIndex(MediaStore.Audio.Media.DATE_MODIFIED)
                val sizeCol = cursor.getColumnIndex(MediaStore.Audio.Media.SIZE)
                val relCol = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    cursor.getColumnIndex(MediaStore.MediaColumns.RELATIVE_PATH)
                } else -1
                val pendCol = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    cursor.getColumnIndex(MediaStore.MediaColumns.IS_PENDING)
                } else -1

                var inspected = 0
                while (cursor.moveToNext() && inspected < 2500) {
                    inspected++
                    if (pendCol >= 0 && cursor.getInt(pendCol) != 0) continue
                    if (idCol < 0 || nameCol < 0) continue

                    val name = cursor.getString(nameCol) ?: continue
                    if (shouldSkipFileName(name)) continue
                    val relPath = if (relCol >= 0) (cursor.getString(relCol) ?: "") else ""
                    val looksWhatsApp = relPath.contains("whatsapp", ignoreCase = true) ||
                        name.startsWith("PTT-", ignoreCase = true) ||
                        name.startsWith("AUD-", ignoreCase = true) ||
                        name.contains("-WA", ignoreCase = true)
                    if (!looksWhatsApp) continue

                    val size = if (sizeCol >= 0) cursor.getLong(sizeCol) else 0L
                    if (size < MIN_AUDIO_BYTES) continue
                    val modSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                    val modMs = if (modSec > 10_000_000_000L) modSec else modSec * 1000L
                    val key = "$name|$size"
                    if (!deduped.containsKey(key)) {
                        val contentUri = ContentUris.withAppendedId(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, cursor.getLong(idCol))
                        deduped[key] = WhatsAppAudioItem(
                            uri = contentUri,
                            displayName = name,
                            categoryLabel = inferCategoryLabel("$relPath/$name"),
                            folderHint = relPath.trimEnd('/').ifBlank { "MediaStore" },
                            sizeBytes = size,
                            lastModifiedMs = modMs,
                            sourceKind = "MediaStore"
                        )
                    }
                }
            }
        } catch (_: Throwable) {
        }

        return sortWhatsAppAudioItems(deduped.values.toList()).take(maxItems)
    }

    fun selectMostRecentSafAudioCandidate(
        items: List<WhatsAppAudioItem>,
        minTimestampMs: Long,
        maxTimestampMs: Long,
        isAlreadyBound: (uriString: String, displayName: String) -> Boolean = { _, _ -> false }
    ): WhatsAppAudioItem? {
        return items
            .asSequence()
            .filter { item ->
                item.sizeBytes >= MIN_AUDIO_BYTES &&
                    !shouldSkipFileName(item.displayName) &&
                    item.lastModifiedMs >= minTimestampMs &&
                    item.lastModifiedMs <= maxTimestampMs &&
                    !isAlreadyBound(item.uri?.toString().orEmpty(), item.displayName)
            }
            .maxWithOrNull(
                compareBy<WhatsAppAudioItem> { it.lastModifiedMs }
                    .thenBy { it.displayName }
            )
    }

    fun findRecentSafAudioUri(
        context: Context,
        minTimestampMs: Long,
        maxTimestampMs: Long = System.currentTimeMillis() + 10_000L,
        isAlreadyBound: (uriString: String, displayName: String) -> Boolean = { _, _ -> false }
    ): Pair<Uri, String>? {
        return try {
            val collected = LinkedHashMap<String, WhatsAppAudioItem>()
            val perms = context.contentResolver.persistedUriPermissions
            for (perm in perms) {
                if (!perm.isReadPermission) continue
                val treeUri = perm.uri
                val rootDocId = DocumentsContract.getTreeDocumentId(treeUri)
                scanDocumentTreeRecursive(
                    context = context,
                    treeUri = treeUri,
                    parentDocId = rootDocId,
                    folderLabel = rootDocId.substringAfterLast('/'),
                    depth = 0,
                    maxDepth = 6,
                    outMap = collected,
                    maxCollect = 160
                )
            }
            val match = selectMostRecentSafAudioCandidate(
                items = collected.values.toList(),
                minTimestampMs = minTimestampMs,
                maxTimestampMs = maxTimestampMs,
                isAlreadyBound = isAlreadyBound
            )
            match?.let { candidate -> candidate.uri?.let { uri -> Pair(uri, candidate.displayName) } }
        } catch (_: Throwable) {
            null
        }
    }

    private fun scanDocumentTreeRecursive(
        context: Context,
        treeUri: Uri,
        parentDocId: String,
        folderLabel: String,
        depth: Int,
        maxDepth: Int,
        outMap: MutableMap<String, WhatsAppAudioItem>,
        maxCollect: Int
    ) {
        if (depth > maxDepth || outMap.size >= maxCollect) return
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId)
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_LAST_MODIFIED
        )
        context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
            val idCol = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameCol = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeCol = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_MIME_TYPE)
            val sizeCol = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_SIZE)
            val modCol = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_LAST_MODIFIED)

            while (cursor.moveToNext() && outMap.size < maxCollect) {
                if (idCol < 0 || nameCol < 0) continue
                val docId = cursor.getString(idCol) ?: continue
                val name = cursor.getString(nameCol) ?: continue
                val mime = if (mimeCol >= 0) (cursor.getString(mimeCol) ?: "") else ""

                if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                    val lowerDir = name.lowercase(Locale.ROOT)
                    val isIrrelevantDir = lowerDir in setOf(
                        "databases", "backups", "stickers", "wallpaper",
                        "whatsapp images", "whatsapp video", "whatsapp documents",
                        "whatsapp animated gifs", "whatsapp profile photos", "whatsapp stickers"
                    )
                    if (!name.startsWith(".") && !isIrrelevantDir) {
                        scanDocumentTreeRecursive(
                            context = context,
                            treeUri = treeUri,
                            parentDocId = docId,
                            folderLabel = name,
                            depth = depth + 1,
                            maxDepth = maxDepth,
                            outMap = outMap,
                            maxCollect = maxCollect
                        )
                    }
                } else {
                    if (shouldSkipFileName(name)) continue
                    val size = if (sizeCol >= 0) cursor.getLong(sizeCol) else 0L
                    if (size < MIN_AUDIO_BYTES) continue
                    val lastMod = if (modCol >= 0) cursor.getLong(modCol) else 0L
                    val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docId)
                    val key = "$name|$size"
                    if (!outMap.containsKey(key)) {
                        outMap[key] = WhatsAppAudioItem(
                            uri = docUri,
                            displayName = name,
                            categoryLabel = inferCategoryLabel("$folderLabel/$name"),
                            folderHint = folderLabel.ifBlank { "Carpeta SAF" },
                            sizeBytes = size,
                            lastModifiedMs = lastMod,
                            sourceKind = "Carpeta SAF"
                        )
                    }
                }
            }
        }
    }
}
