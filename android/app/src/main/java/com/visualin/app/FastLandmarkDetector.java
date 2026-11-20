package com.visualin.app;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;
import androidx.annotation.Keep;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Low-latency MediaPipe hand and upper-body tracker using RunningMode.VIDEO,
 * direct bulk YUV-to-RGB conversion into reusable direct ByteBuffers (avoiding
 * per-frame Bitmap allocations/recycling and JPEG round-trips), concurrent
 * pose/hand execution when both run on a frame, and packed double[] JNI output.
 */
@Keep
public final class FastLandmarkDetector implements AutoCloseable {
    private static final String TAG = "VisualinFast";
    private static final int[] POSE_INDICES = {0, 11, 12, 13, 14, 15, 16, 7, 8, 23, 24};
    private static final int HAND_LANDMARKS = 21;

    private final Context appContext;
    private HandLandmarker handLandmarker;
    private PoseLandmarker poseLandmarker;
    private ExecutorService poseExecutor;

    private long lastHandTimestampMs = 0L;
    private long lastPoseTimestampMs = 0L;
    private int lastRotation = -1;
    private int lastFrameWidth = 0;
    private int lastFrameHeight = 0;
    private ImageProcessingOptions cachedImageOptions;

    private byte[] yBytes = new byte[0];
    private byte[] uBytes = new byte[0];
    private byte[] vBytes = new byte[0];

    private byte[] handRgbBytes = new byte[0];
    private ByteBuffer handRgbBuffer;
    private int handRgbStride = 0;

    private byte[] poseRgbBytes = new byte[0];
    private ByteBuffer poseRgbBuffer;

    private boolean closed = false;

    @Keep
    public FastLandmarkDetector(Context context) {
        this.appContext = context.getApplicationContext();
        this.handLandmarker = createHandLandmarker();
    }

    private HandLandmarker createHandLandmarker() {
        BaseOptions base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(Delegate.CPU)
            .build();
        HandLandmarker.HandLandmarkerOptions options =
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.45f)
                .setMinTrackingConfidence(0.45f)
                .build();
        return HandLandmarker.createFromOptions(appContext, options);
    }

    private PoseLandmarker ensurePoseLandmarker() {
        if (poseLandmarker == null) {
            BaseOptions base = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .setDelegate(Delegate.CPU)
                .build();
            PoseLandmarker.PoseLandmarkerOptions options =
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(base)
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumPoses(1)
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinPosePresenceConfidence(0.45f)
                    .setMinTrackingConfidence(0.45f)
                    .build();
            poseLandmarker = PoseLandmarker.createFromOptions(appContext, options);
        }
        if (poseExecutor == null) {
            poseExecutor = Executors.newSingleThreadExecutor();
        }
        return poseLandmarker;
    }

    @Keep
    public void releasePose() {
        if (poseExecutor != null) {
            poseExecutor.shutdownNow();
            poseExecutor = null;
        }
        if (poseLandmarker != null) {
            poseLandmarker.close();
            poseLandmarker = null;
        }
        poseRgbBuffer = null;
        poseRgbBytes = new byte[0];
    }

    /**
     * Returns packed landmarks:
     * [poseSampled (0/1), poseError (0/1), poseCount (0 or 11), ...pose (x,y),
     *  handCount (0..2), ...hands (21 * (x,y,z))]
     */
    @Keep
    public double[] detect(
        ByteBuffer y,
        ByteBuffer u,
        ByteBuffer v,
        int yLen,
        int uLen,
        int vLen,
        int width,
        int height,
        int yStride,
        int uStride,
        int vStride,
        int uPixel,
        int vPixel,
        int rotation,
        int runPoseFlag
    ) {
        if (closed || handLandmarker == null) {
            throw new IllegalStateException("FastLandmarkDetector is closed");
        }
        final boolean runPose = runPoseFlag != 0;
        final int normRot = ((rotation % 360) + 360) % 360;
        if (lastFrameWidth != 0 &&
            (lastFrameWidth != width || lastFrameHeight != height || lastRotation != normRot)) {
            handLandmarker.close();
            handLandmarker = createHandLandmarker();
            if (poseLandmarker != null) {
                poseLandmarker.close();
                poseLandmarker = null;
            }
        }
        lastFrameWidth = width;
        lastFrameHeight = height;

        final int copiedY = copyBuffer(y, yLen, yBytesRef(yLen));
        final int copiedU = copyBuffer(u, uLen, uBytesRef(uLen));
        final int copiedV = copyBuffer(v, vLen, vBytesRef(vLen));

        final int rowStride = ((width * 3 + 3) & ~3);
        final int rgbSize = rowStride * height;
        handRgbStride = rowStride;
        if (handRgbBytes.length != rgbSize) {
            handRgbBytes = new byte[rgbSize];
        }
        if (handRgbBuffer == null || handRgbBuffer.capacity() != rgbSize) {
            handRgbBuffer = ByteBuffer.allocateDirect(rgbSize);
        }

        convertYuvToRgbBytes(
            width, height,
            copiedY, copiedU, copiedV,
            yStride, uStride, vStride, uPixel, vPixel,
            rowStride
        );

        if (cachedImageOptions == null || lastRotation != normRot) {
            cachedImageOptions = ImageProcessingOptions.builder()
                .setRotationDegrees(normRot)
                .build();
            lastRotation = normRot;
        }

        Future<double[]> poseFuture = null;
        boolean poseError = false;
        if (runPose) {
            try {
                final PoseLandmarker pose = ensurePoseLandmarker();
                poseFuture = poseExecutor.submit(
                    () -> detectPoseFromRgbBytes(pose, width, height, rowStride, normRot)
                );
            } catch (Throwable t) {
                Log.e(TAG, "Pose submit error", t);
                poseError = true;
            }
        }

        handRgbBuffer.clear();
        handRgbBuffer.put(handRgbBytes, 0, rgbSize);
        handRgbBuffer.rewind();

        List<List<NormalizedLandmark>> detectedHands;
        MPImage handImage = new ByteBufferImageBuilder(
            handRgbBuffer, width, height, MPImage.IMAGE_FORMAT_RGB
        ).build();
        try {
            long nowMs = SystemClock.uptimeMillis();
            if (nowMs <= lastHandTimestampMs) {
                nowMs = lastHandTimestampMs + 1;
            }
            lastHandTimestampMs = nowMs;
            HandLandmarkerResult handResult = handLandmarker.detectForVideo(
                handImage, cachedImageOptions, nowMs
            );
            detectedHands = handResult.landmarks();
        } finally {
            handImage.close();
        }

        double[] posePoints = null;
        if (poseFuture != null) {
            try {
                posePoints = poseFuture.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                poseError = true;
            } catch (ExecutionException e) {
                Log.e(TAG, "Pose future error", e);
                poseError = true;
            }
        }

        final int poseCount = posePoints != null ? posePoints.length / 2 : 0;
        final int handCount = detectedHands != null ? detectedHands.size() : 0;

        final double[] out = new double[4 + poseCount * 2 + handCount * HAND_LANDMARKS * 3];
        out[0] = runPose ? 1.0 : 0.0;
        out[1] = poseError ? 1.0 : 0.0;
        out[2] = poseCount;
        int cursor = 3;
        if (posePoints != null) {
            System.arraycopy(posePoints, 0, out, cursor, posePoints.length);
            cursor += posePoints.length;
        }
        out[cursor++] = handCount;
        if (detectedHands != null) {
            for (int h = 0; h < handCount; h++) {
                List<NormalizedLandmark> hand = detectedHands.get(h);
                final int count = Math.min(HAND_LANDMARKS, hand.size());
                for (int i = 0; i < count; i++) {
                    NormalizedLandmark lm = hand.get(i);
                    out[cursor + i * 3] = lm.x();
                    out[cursor + i * 3 + 1] = lm.y();
                    out[cursor + i * 3 + 2] = lm.z();
                }
                cursor += HAND_LANDMARKS * 3;
            }
        }
        return out;
    }

    private double[] detectPoseFromRgbBytes(
        PoseLandmarker pose,
        int width,
        int height,
        int srcRowStride,
        int rotation
    ) {
        final int step = Math.max(1, (Math.max(width, height) + 383) / 384);
        final int sampleW = width / step;
        final int sampleH = height / step;
        final boolean swapDims = rotation == 90 || rotation == 270;
        final int targetW = swapDims ? sampleH : sampleW;
        final int targetH = swapDims ? sampleW : sampleH;
        final int dstRowStride = ((targetW * 3 + 3) & ~3);
        final int needed = dstRowStride * targetH;
        if (poseRgbBytes.length != needed) {
            poseRgbBytes = new byte[needed];
        }
        if (poseRgbBuffer == null || poseRgbBuffer.capacity() != needed) {
            poseRgbBuffer = ByteBuffer.allocateDirect(needed);
        }

        final byte[] dst = poseRgbBytes;
        final byte[] src = handRgbBytes;
        final int stepBytes = step * 3;

        if (rotation == 0) {
            if (step == 1 && srcRowStride == dstRowStride) {
                System.arraycopy(src, 0, dst, 0, needed);
            } else {
                for (int sy = 0; sy < sampleH; sy++) {
                    int sIdx = sy * step * srcRowStride;
                    int dIdx = sy * dstRowStride;
                    for (int sx = 0; sx < sampleW; sx++) {
                        dst[dIdx] = src[sIdx];
                        dst[dIdx + 1] = src[sIdx + 1];
                        dst[dIdx + 2] = src[sIdx + 2];
                        sIdx += stepBytes;
                        dIdx += 3;
                    }
                }
            }
        } else if (rotation == 90) {
            for (int sy = 0; sy < sampleH; sy++) {
                int sIdx = sy * step * srcRowStride;
                final int dx = sampleH - 1 - sy;
                int dBase = dx * 3;
                for (int sx = 0; sx < sampleW; sx++) {
                    dst[dBase] = src[sIdx];
                    dst[dBase + 1] = src[sIdx + 1];
                    dst[dBase + 2] = src[sIdx + 2];
                    sIdx += stepBytes;
                    dBase += dstRowStride;
                }
            }
        } else if (rotation == 180) {
            for (int sy = 0; sy < sampleH; sy++) {
                int sIdx = sy * step * srcRowStride;
                int dIdx = (sampleH - 1 - sy) * dstRowStride + (sampleW - 1) * 3;
                for (int sx = 0; sx < sampleW; sx++) {
                    dst[dIdx] = src[sIdx];
                    dst[dIdx + 1] = src[sIdx + 1];
                    dst[dIdx + 2] = src[sIdx + 2];
                    sIdx += stepBytes;
                    dIdx -= 3;
                }
            }
        } else {
            // 270 degrees CW
            for (int sy = 0; sy < sampleH; sy++) {
                int sIdx = sy * step * srcRowStride;
                int dIdx = (sampleW - 1) * dstRowStride + sy * 3;
                for (int sx = 0; sx < sampleW; sx++) {
                    dst[dIdx] = src[sIdx];
                    dst[dIdx + 1] = src[sIdx + 1];
                    dst[dIdx + 2] = src[sIdx + 2];
                    sIdx += stepBytes;
                    dIdx -= dstRowStride;
                }
            }
        }

        poseRgbBuffer.clear();
        poseRgbBuffer.put(dst, 0, needed);
        poseRgbBuffer.rewind();

        MPImage poseImage = new ByteBufferImageBuilder(
            poseRgbBuffer, targetW, targetH, MPImage.IMAGE_FORMAT_RGB
        ).build();
        try {
            long nowMs = SystemClock.uptimeMillis();
            if (nowMs <= lastPoseTimestampMs) {
                nowMs = lastPoseTimestampMs + 1;
            }
            lastPoseTimestampMs = nowMs;
            PoseLandmarkerResult result = pose.detectForVideo(poseImage, nowMs);
            if (result.landmarks().isEmpty()) {
                return null;
            }
            List<NormalizedLandmark> landmarks = result.landmarks().get(0);
            double[] points = new double[POSE_INDICES.length * 2];
            for (int i = 0; i < POSE_INDICES.length; i++) {
                NormalizedLandmark lm = landmarks.get(POSE_INDICES[i]);
                points[2 * i] = lm.x();
                points[2 * i + 1] = lm.y();
            }
            return points;
        } finally {
            poseImage.close();
        }
    }

    private void convertYuvToRgbBytes(
        int width,
        int height,
        int yLen,
        int uLen,
        int vLen,
        int yStride,
        int uStride,
        int vStride,
        int uPixel,
        int vPixel,
        int rgbRowStride
    ) {
        final byte[] rgb = handRgbBytes;
        final byte[] yArr = yBytes;
        final byte[] uArr = uBytes;
        final byte[] vArr = vBytes;
        final int uvLimit = Math.min(uLen, vLen);

        for (int row = 0; row < height; row++) {
            final int yRow = row * yStride;
            final int uvRow = row >> 1;
            final int uRowBase = uvRow * uStride;
            final int vRowBase = uvRow * vStride;
            int outIdx = row * rgbRowStride;

            int col = 0;
            for (; col + 1 < width; col += 2) {
                final int halfCol = col >> 1;
                final int uIdx = uRowBase + halfCol * uPixel;
                final int vIdx = vRowBase + halfCol * vPixel;
                final int cb = (uIdx < uvLimit ? (uArr[uIdx] & 255) : 128) - 128;
                final int cr = (vIdx < uvLimit ? (vArr[vIdx] & 255) : 128) - 128;
                final int rDelta = 409 * cr + 128;
                final int gDelta = -100 * cb - 208 * cr + 128;
                final int bDelta = 516 * cb + 128;

                int yIdx0 = yRow + col;
                int y0 = yIdx0 < yLen ? (yArr[yIdx0] & 255) : 0;
                int luma0 = y0 > 16 ? (y0 - 16) * 298 : 0;
                rgb[outIdx] = clampByte((luma0 + rDelta) >> 8);
                rgb[outIdx + 1] = clampByte((luma0 + gDelta) >> 8);
                rgb[outIdx + 2] = clampByte((luma0 + bDelta) >> 8);

                int yIdx1 = yIdx0 + 1;
                int y1 = yIdx1 < yLen ? (yArr[yIdx1] & 255) : 0;
                int luma1 = y1 > 16 ? (y1 - 16) * 298 : 0;
                rgb[outIdx + 3] = clampByte((luma1 + rDelta) >> 8);
                rgb[outIdx + 4] = clampByte((luma1 + gDelta) >> 8);
                rgb[outIdx + 5] = clampByte((luma1 + bDelta) >> 8);
                outIdx += 6;
            }
            if (col < width) {
                final int halfCol = col >> 1;
                final int uIdx = uRowBase + halfCol * uPixel;
                final int vIdx = vRowBase + halfCol * vPixel;
                final int cb = (uIdx < uvLimit ? (uArr[uIdx] & 255) : 128) - 128;
                final int cr = (vIdx < uvLimit ? (vArr[vIdx] & 255) : 128) - 128;
                int yIdx = yRow + col;
                int yVal = yIdx < yLen ? (yArr[yIdx] & 255) : 0;
                int luma = yVal > 16 ? (yVal - 16) * 298 : 0;
                rgb[outIdx] = clampByte((luma + 409 * cr + 128) >> 8);
                rgb[outIdx + 1] = clampByte((luma - 100 * cb - 208 * cr + 128) >> 8);
                rgb[outIdx + 2] = clampByte((luma + 516 * cb + 128) >> 8);
            }
        }
    }

    private static byte clampByte(int v) {
        if (v <= 0) return 0;
        if (v >= 255) return (byte) 255;
        return (byte) v;
    }

    private byte[] yBytesRef(int needed) {
        if (yBytes.length < needed) yBytes = new byte[needed];
        return yBytes;
    }

    private byte[] uBytesRef(int needed) {
        if (uBytes.length < needed) uBytes = new byte[needed];
        return uBytes;
    }

    private byte[] vBytesRef(int needed) {
        if (vBytes.length < needed) vBytes = new byte[needed];
        return vBytes;
    }

    private static int copyBuffer(ByteBuffer buf, int len, byte[] dst) {
        buf.rewind();
        final int safeLen = Math.min(len, Math.min(buf.remaining(), dst.length));
        buf.get(dst, 0, safeLen);
        return safeLen;
    }

    @Override
    @Keep
    public void close() {
        closed = true;
        releasePose();
        if (handLandmarker != null) {
            handLandmarker.close();
            handLandmarker = null;
        }
        handRgbBuffer = null;
        handRgbBytes = new byte[0];
    }
}
