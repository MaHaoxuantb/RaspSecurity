import numpy as np
from PIL import Image, ImageFilter

class Rest:
    # Adaptive background (kept in memory across calls)
    _bg_small: np.ndarray | None = None

    async def detectMovement(image1, image2) -> float:
        """Return motion score as a changed-pixel ratio (0.0 ~ 1.0).

        Implements: downsample + grayscale + blur + thresholded change ratio
        against an adaptive background.

        Notes:
        - Keeps its own adaptive background; `image1`/`image2` are accepted to
          avoid changing the rest of your code. The first frame seen initializes
          the background.
        """
        print("Important: Rest.detectMovement is deprecated.")

        # Tunables (keep these small and boring)
        target_size = (160, 90)   # (W, H) downsample for stability + speed
        blur_radius = 1.5         # blur to suppress MJPEG noise + tiny jitter
        diff_threshold = 20       # pixel intensity threshold (0-255)
        alpha = 0.02              # background update speed

        # Use the *current* frame (your call-site passes (empty_frame, now_frame))
        frame = image2

        # 1) Downsample + 2) Grayscale + 3) Blur
        pil = Image.fromarray(frame)
        pil = pil.resize(target_size, Image.BILINEAR).convert("L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        cur = np.asarray(pil, dtype=np.float32)

        # Initialize background on first use
        if Rest._bg_small is None:
            Rest._bg_small = cur.copy()
            return 0.0

        bg = Rest._bg_small

        # 4) Motion as changed-pixel ratio against background
        diff = np.abs(cur - bg)
        changed = diff > float(diff_threshold)
        motion_ratio = float(np.mean(changed))  # 0.0 ~ 1.0

        # Adaptive background update:
        # - Update faster when scene is stable
        # - Update slower during motion so moving objects don't get absorbed instantly
        alpha_use = alpha if motion_ratio < 0.02 else (alpha * 0.2)
        Rest._bg_small = (1.0 - alpha_use) * bg + alpha_use * cur

        return motion_ratio
