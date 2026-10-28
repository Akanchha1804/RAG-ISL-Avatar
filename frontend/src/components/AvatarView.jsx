// The 3D avatar renderer (three.js + VRM retargeting) was removed on purpose
// in the animation reset: the motion never converged, so the whole pipeline is
// being rebuilt from scratch. This component is a deliberately inert
// placeholder so the page layout stays stable while that happens. It must not
// animate, and it must not imply that anything is being signed.
export default function AvatarView() {
  return (
    <div className="flex w-full flex-col items-center gap-2">
      <div className="flex aspect-[4/3] w-full max-w-[480px] items-center justify-center rounded-lg border border-dashed border-white/15 bg-slate-900/60">
        <span className="px-6 text-center text-xs leading-relaxed text-slate-400">
          Avatar animation removed.
          <br />
          Rebuilding the sign renderer from scratch.
        </span>
      </div>
    </div>
  );
}
