package com.visualin.app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.SystemClock;
import androidx.annotation.Keep;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.nio.ByteBuffer;

/** Owned by the camera worker. Coordinates refer to the physically upright bitmap. */
@Keep
public final class UpperBodyDetector implements AutoCloseable {
    private final PoseLandmarker detector;
    private byte[] yBytes = new byte[0];
    private byte[] uBytes = new byte[0];
    private byte[] vBytes = new byte[0];
    private int[] pixelBuffer = new int[0];
    private long lastTimestampMs = 0L;

    public UpperBodyDetector(Context context) {
        BaseOptions base = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .setDelegate(Delegate.CPU)
            .build();
        detector = PoseLandmarker.createFromOptions(context,
            PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.45f)
                .setMinTrackingConfidence(0.45f)
                .build());
    }

    public double[] detect(ByteBuffer y, ByteBuffer u, ByteBuffer v, int width, int height,
            int yStride, int uStride, int vStride, int uPixel, int vPixel, int rotation) {
        int step = Math.max(1, (Math.max(width, height) + 639) / 640);
        int w = width / step, h = height / step;
        int needed = w * h;
        if (pixelBuffer.length != needed) {
            pixelBuffer = new int[needed];
        }
        final int yLen = copyBuffer(y, 0);
        final int uLen = copyBuffer(u, 1);
        final int vLen = copyBuffer(v, 2);
        final byte[] ya = yBytes, ua = uBytes, va = vBytes;
        int[] pixels = pixelBuffer;
        for (int row = 0; row < h; row++) {
            int sy = row * step;
            int yRow = sy * yStride;
            int uRow = (sy >> 1) * uStride;
            int vRow = (sy >> 1) * vStride;
            int pRow = row * w;
            for (int col = 0; col < w; col++) {
                int sx = col * step;
                int yIdx = yRow + sx;
                int uIdx = uRow + (sx >> 1) * uPixel;
                int vIdx = vRow + (sx >> 1) * vPixel;
                int yv = yIdx < yLen ? (ya[yIdx] & 255) : 0;
                int luma = yv > 16 ? (yv - 16) * 298 : 0;
                int cb = (uIdx < uLen ? (ua[uIdx] & 255) : 128) - 128;
                int cr = (vIdx < vLen ? (va[vIdx] & 255) : 128) - 128;
                pixels[pRow + col] = 0xff000000 | channel(luma + 409 * cr) << 16
                    | channel(luma - 100 * cb - 208 * cr) << 8 | channel(luma + 516 * cb);
            }
        }
        Bitmap sensor = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888);
        Bitmap upright = sensor;
        if (rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            upright = Bitmap.createBitmap(sensor, 0, 0, w, h, matrix, true);
        }
        MPImage image = new BitmapImageBuilder(upright).build();
        try {
            long nowMs = SystemClock.uptimeMillis();
            if (nowMs <= lastTimestampMs) nowMs = lastTimestampMs + 1;
            lastTimestampMs = nowMs;
            PoseLandmarkerResult result = detector.detectForVideo(image, nowMs);
            if (result.landmarks().isEmpty()) return new double[0];
            int[] indices = {0, 11, 12, 13, 14, 15, 16, 7, 8, 23, 24};
            double[] points = new double[indices.length * 2];
            for (int i = 0; i < indices.length; i++) {
                points[2*i] = result.landmarks().get(0).get(indices[i]).x();
                points[2*i+1] = result.landmarks().get(0).get(indices[i]).y();
            }
            return points;
        } finally {
            image.close();
            if (upright != sensor) upright.recycle();
            sensor.recycle();
        }
    }

    private int copyBuffer(ByteBuffer buf, int slot) {
        buf.rewind();
        int len = buf.remaining();
        if (slot == 0) {
            if (yBytes.length < len) yBytes = new byte[len];
            buf.get(yBytes, 0, len);
        } else if (slot == 1) {
            if (uBytes.length < len) uBytes = new byte[len];
            buf.get(uBytes, 0, len);
        } else {
            if (vBytes.length < len) vBytes = new byte[len];
            buf.get(vBytes, 0, len);
        }
        return len;
    }

    private static int channel(int value) {
        return Math.max(0, Math.min(255, (value + 128) >> 8));
    }

    @Override public void close() { detector.close(); }
}
