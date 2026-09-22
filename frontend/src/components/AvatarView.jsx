import { useEffect, useRef, useState } from 'react';

export default function AvatarView({ landmarkUrl }) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState('No landmark data');
  const [frameData, setFrameData] = useState(null);
  const animRef = useRef(null);

  useEffect(() => {
    if (!landmarkUrl) {
      setStatus('No landmark data');
      setFrameData(null);
      return;
    }

    setStatus('Loading landmarks...');
    fetch(landmarkUrl)
      .then(res => res.json())
      .then(data => {
        setFrameData(data);
        setStatus(`Loaded ${data.processed_frames} frames`);
      })
      .catch(err => {
        setStatus('Failed to load landmarks');
        console.error(err);
      });
  }, [landmarkUrl]);

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let frameIndex = 0;
    const fps = frameData.fps || 25;
    const interval = 1000 / fps;

    const drawFrame = () => {
      const frames = frameData.frames;
      if (!frames || frames.length === 0) return;

      frameIndex = frameIndex % frames.length;
      const frame = frames[frameIndex];

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#1a1a2e';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (frame.hands && frame.hands.length > 0) {
        frame.hands.forEach((hand, handIdx) => {
          const color = hand.handedness === 'Left' ? '#00d4ff' : '#ff6b6b';
          const offsetX = handIdx * 120;

          ctx.strokeStyle = color;
          ctx.lineWidth = 1;

          const connections = [
            [0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],
            [0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20],
            [5,9],[9,13],[13,17]
          ];

          connections.forEach(([a, b]) => {
            const la = hand.landmarks[a];
            const lb = hand.landmarks[b];
            if (la && lb) {
              ctx.beginPath();
              ctx.moveTo(la.x * 200 + offsetX + 50, la.y * 200 + 30);
              ctx.lineTo(lb.x * 200 + offsetX + 50, lb.y * 200 + 30);
              ctx.stroke();
            }
          });

          hand.landmarks.forEach((lm, i) => {
            ctx.beginPath();
            ctx.arc(lm.x * 200 + offsetX + 50, lm.y * 200 + 30, 3, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#fff' : color;
            ctx.fill();
          });
        });
      }

      ctx.fillStyle = '#ffffff';
      ctx.font = '12px monospace';
      ctx.fillText(`Frame: ${frameIndex + 1}/${frames.length}`, 10, canvas.height - 10);
      ctx.fillText(`FPS: ${fps}`, 10, canvas.height - 25);

      frameIndex++;
    };

    animRef.current = setInterval(drawFrame, interval);
    drawFrame();

    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, [frameData]);

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={300}
        height={250}
        className="border border-gray-300 rounded-lg bg-gray-900"
      />
      <span className="text-xs text-gray-500">{status}</span>
    </div>
  );
}
