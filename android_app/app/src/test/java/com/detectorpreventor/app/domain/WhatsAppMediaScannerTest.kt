package com.detectorpreventor.app.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class WhatsAppMediaScannerTest {

    @Test
    fun testSupportedWhatsAppAudioExtensions() {
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("PTT-20261109-WA0001.opus"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("AUD-20261109-WA0002.M4A"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("nota_voz.ogg"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("audio_banco.mp3"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("grabacion.wav"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("clip.aac"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("voz.amr"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("voz.flac"))
        assertTrue(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("voz.3ga"))

        assertFalse(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension(".nomedia"))
        assertFalse(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("foto.jpg"))
        assertFalse(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("video.mp4"))
        assertFalse(WhatsAppMediaScanner.isSupportedWhatsAppAudioExtension("sin_extension"))
    }

    @Test
    fun testShouldSkipFileNameFiltersHiddenPendingAndNonAudio() {
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName(""))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("   "))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName(".nomedia"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName(".hidden.opus"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("PTT-2026-WA0001.opus.tmp"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("PTT-2026-WA0001.opus.pending"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("msg.crypt14.enc"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("download.part"))
        assertTrue(WhatsAppMediaScanner.shouldSkipFileName("imagen.png"))

        assertFalse(WhatsAppMediaScanner.shouldSkipFileName("PTT-20261109-WA0014.opus"))
        assertFalse(WhatsAppMediaScanner.shouldSkipFileName("AUD-20261109-WA0003.m4a"))
    }

    @Test
    fun testInferCategoryLabel() {
        assertEquals(
            "Nota de Voz",
            WhatsAppMediaScanner.inferCategoryLabel("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/202645/PTT-20261109-WA0001.opus")
        )
        assertEquals(
            "Audio WhatsApp",
            WhatsAppMediaScanner.inferCategoryLabel("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/AUD-20261109-WA0004.m4a")
        )
        assertEquals(
            "WhatsApp Business",
            WhatsAppMediaScanner.inferCategoryLabel("/storage/emulated/0/Android/media/com.whatsapp.w4b/WhatsApp Business/Media/record.ogg")
        )
        assertEquals(
            "Audio Local",
            WhatsAppMediaScanner.inferCategoryLabel("/storage/emulated/0/Download/grabacion.wav")
        )
    }

    @Test
    fun testSortWhatsAppAudioItemsDescendingByDate() {
        val older = WhatsAppAudioItem(
            uri = null,
            displayName = "PTT-older.opus",
            categoryLabel = "Nota de Voz",
            folderHint = "202644",
            sizeBytes = 14_000L,
            lastModifiedMs = 1_700_000_000_000L,
            sourceKind = "Archivo Directo"
        )
        val newer = older.copy(
            displayName = "PTT-newer.opus",
            lastModifiedMs = 1_700_000_500_000L
        )
        val sorted = WhatsAppMediaScanner.sortWhatsAppAudioItems(listOf(older, newer))
        assertEquals(listOf("PTT-newer.opus", "PTT-older.opus"), sorted.map { it.displayName })
        assertTrue(newer.formattedSize.contains("KB"))
    }

    @Test
    fun testCandidateWhatsAppRootsIncludesAndroidMediaAndLegacyPaths() {
        val roots = WhatsAppMediaScanner.candidateWhatsAppRoots().map { it.path.replace('\\', '/') }
        assertTrue(roots.any { it.contains("Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes") })
        assertTrue(roots.any { it.contains("Android/media/com.whatsapp.w4b/WhatsApp Business/Media/WhatsApp Voice Notes") })
        assertTrue(roots.any { it.endsWith("WhatsApp/Media/WhatsApp Voice Notes") })
    }

    @Test
    fun testSelectMostRecentSafAudioCandidateRespectsWindowAndBoundFilter() {
        val baseTime = 1_715_000_000_000L
        val tooOld = WhatsAppAudioItem(
            uri = null,
            displayName = "PTT-old.opus",
            categoryLabel = "Nota de Voz",
            folderHint = "202645",
            sizeBytes = 12_000L,
            lastModifiedMs = baseTime - 60_000L,
            sourceKind = "SAF"
        )
        val tooSmall = tooOld.copy(
            displayName = "PTT-tiny.opus",
            sizeBytes = 64L,
            lastModifiedMs = baseTime + 2_000L
        )
        val alreadyBound = tooOld.copy(
            displayName = "PTT-bound.opus",
            lastModifiedMs = baseTime + 5_000L
        )
        val validRecent = tooOld.copy(
            displayName = "PTT-valid.opus",
            lastModifiedMs = baseTime + 3_000L
        )
        val pendingTmp = tooOld.copy(
            displayName = "PTT-pending.opus.tmp",
            lastModifiedMs = baseTime + 4_000L
        )

        val selected = WhatsAppMediaScanner.selectMostRecentSafAudioCandidate(
            items = listOf(tooOld, tooSmall, alreadyBound, validRecent, pendingTmp),
            minTimestampMs = baseTime - 15_000L,
            maxTimestampMs = baseTime + 10_000L,
            isAlreadyBound = { _, name -> name == "PTT-bound.opus" }
        )
        assertEquals("PTT-valid.opus", selected?.displayName)
    }
}

