import { useEffect, useMemo, useRef, useState } from 'react';
import VrmAvatar from './VrmAvatar';

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

// Landmarks are normalized image coordinates (~0-1). Fixed-scale drawing used
// to push the right hand off a 300px canvas, so the view is fit to the data
// bounds instead - computed once per dataset to keep the framing stable.
function computeBounds(frames) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const frame of frames || []) {
    for (const hand of frame.hands || []) {
      for (const lm of hand.landmarks || []) {
        if (lm.x < minX) minX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y > maxY) maxY = lm.y;
      }
    }
  }

  if (!Number.isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}

function makeProjector(bounds, width, height) {
  const pad = 18;
  const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
  const offsetX = pad + (width - pad * 2 - spanX * scale) / 2 - bounds.minX * scale;
  const offsetY = pad + (height - pad * 2 - spanY * scale) / 2 - bounds.minY * scale;
  return (lm) => ({ x: lm.x * scale + offsetX, y: lm.y * scale + offsetY });
}

export default function AvatarView({ landmarkUrl, animation }) {
  const canvasRef = useRef(null);
  // { url, data, bounds, status, error } - keyed by the URL it belongs to so
  // a stale response can never overwrite a newer one.
  const [load, setLoad] = useState(null);
  // If the VRM avatar cannot load, latch into the 2D skeleton / clips
  // fallback (a missing model file will not heal between translations).
  const [vrmFailed, setVrmFailed] = useState(false);

  const clips = useMemo(
    () =>
      Array.isArray(animation?.clip_playlist)
        ? animation.clip_playlist.filter((c) => c && c.landmark_clip_url)
        : [],
    [animation],
  );

  useEffect(() => {
    if (!landmarkUrl) return undefined;

    const controller = new AbortController();

    fetch(landmarkUrl, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${landmarkUrl}`);
        return res.json();
      })
      .then((data) => {
        if (!Array.isArray(data.frames) || data.frames.length === 0) {
          throw new Error('landmark file contains no frames');
        }
        const bounds = computeBounds(data.frames);
        const hands = data.frames.reduce((n, f) => n + (f.hands?.length || 0), 0);
        setLoad({
          url: landmarkUrl,
          data,
          bounds,
          status: !bounds
            ? 'Landmark file has no hand data'
            : `Loaded ${data.frames.length} frames (${hands} hand frames)`,
        });
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        console.error(err);
        setLoad({ url: landmarkUrl, error: `Failed to load landmarks: ${err.message}` });
      });

    return () => controller.abort();
  }, [landmarkUrl]);

  const active = load && load.url === landmarkUrl ? load : null;

  const status = useMemo(() => {
    if (landmarkUrl) {
      if (!active) return 'Loading landmarks...';
      return active.status || active.error || '';
    }
    if (clips.length > 0) return `Sign clips: ${clips.length}`;
    return 'No animation available';
  }, [landmarkUrl, active, clips.length]);

  const data = active?.bounds ? active.data : null;
  const bounds = active?.bounds || null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!data || !bounds || !canvas) return undefined;

    const frames = data.frames;
    const ctx = canvas.getContext('2d');
    const project = makeProjector(bounds, canvas.width, canvas.height);
    const fps = data.fps || 25;
    const frameDuration = 1000 / fps;

    let frameIndex = 0;
    let lastTick = 0;
    let rafId = null;

    const drawFrame = (index) => {
      const frame = frames[index % frames.length];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      (frame.hands || []).forEach((hand, handIdx) => {
        const color =
          hand.handedness === 'Left'
            ? '#00d4ff'
            : hand.handedness === 'Right'
              ? '#ff6b6b'
              : handIdx === 0
                ? '#00d4ff'
                : '#ff6b6b';
        const points = (hand.landmarks || []).map(project);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        for (const [a, b] of CONNECTIONS) {
          const pa = points[a];
          const pb = points[b];
          if (!pa || !pb) continue;
          ctx.beginPath();
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
        }

        points.forEach((p, i) => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, i === 0 ? 4 : 3, 0, Math.PI * 2);
          ctx.fillStyle = i === 0 ? '#ffffff' : color;
          ctx.fill();
        });
      });

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '12px monospace';
      ctx.fillText(`Frame: ${(index % frames.length) + 1}/${frames.length}`, 8, 16);
      ctx.fillText(`FPS: ${fps}`, 8, 32);
    };

    // rAF keeps playback smooth; setInterval drifted and the handle was not
    // reliably cancelled on re-render.
    const tick = (now) => {
      if (!lastTick) lastTick = now;
      if (now - lastTick >= frameDuration) {
        frameIndex += 1;
        lastTick = now;
        drawFrame(frameIndex);
      }
      rafId = requestAnimationFrame(tick);
    };

    drawFrame(0);
    rafId = requestAnimationFrame(tick);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [data, bounds]);

  // Primary path: the project's own VRM avatar is always on screen.
  // It signs the sentence landmark sequence when one exists, and
  // otherwise signs the per-gloss landmark clips in sequence.
  // Only the avatar performs - there are no video clips in this view.
  if (!vrmFailed) {
    return (
      <div className="flex w-full flex-col items-center gap-2">
        <VrmAvatar
          landmarkUrl={landmarkUrl}
          playlistUrls={landmarkUrl ? [] : clips.map((c) => c.landmark_clip_url)}
          onError={() => setVrmFailed(true)}
        />
        {!landmarkUrl && clips.length === 0 && (
          <span className="text-xs text-slate-400">No animation available</span>
        )}
      </div>
    );
  }

  // Fallback path (VRM could not load): corpus sentences keep the
  // frame-by-frame skeleton animation below.
  if (!landmarkUrl) {
    return (
      <div className="flex w-full flex-col items-center gap-2">
        <span className="text-xs text-slate-400">Avatar unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex w-full flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={480}
        height={360}
        role="img"
        aria-label="Indian Sign Language hand landmark animation"
        className="border border-white/10 rounded-lg bg-slate-900 w-full max-w-[480px] h-auto"
      />
      <span className="text-xs text-slate-400">{status}</span>
    </div>
  );
}
