using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Hand/finger retarget audit by SINGLE-JOINT PROBING.
///
/// The previous version of this harness measured aggregate curl and produced
/// numbers that were indistinguishable from its own bugs (tipless Distal
/// bones inherited their parent's direction, so Distal rows simply echoed
/// Intermediate rows; and a degenerate synthetic hand made every joint read
/// as "backward"). Aggregate statistics cannot localise a mapping error, so
/// this version tests the one thing that can: DRIVE ISOLATION.
///
/// METHOD
/// ------
/// For each of the 15 joints per side, build a synthetic hand that is dead
/// straight everywhere EXCEPT that one joint, which is flexed by a known
/// angle. Then ask the avatar which bones actually rotated.
///
/// Correct mapping  -> exactly ONE bone moves, and it is the bone that
///                     landmark segment is supposed to drive.
/// Off-by-one       -> the neighbouring bone moves instead.
/// Wrong bone       -> some unrelated bone moves.
/// Dead bone        -> NOTHING moves (the pre-fix failure mode: Distal bones
///                     had no rest reference and silently never animated).
///
/// Because each probe isolates one joint, a failure names the defect exactly
/// instead of producing an average.
///
/// Rotation is measured as a quaternion delta from the captured rest pose,
/// so it works for tipless Distal bones (no child transform needed).
///
/// Runs automatically after scene load; no setup. Writes
/// bone_audit_report.txt next to the project and logs to the Console.
/// </summary>
public class BoneAudit : MonoBehaviour
{
    private const float PROBE_FLEX_DEG = 40f;   // obvious but within any sane limit
    private const float MOVED_EPS_DEG = 5f;      // above smoothing noise
    private const int SETTLE_FRAMES = 45;        // let smoothing converge
    private const float PROBE_HOLD_FPS = 60f;    // single-frame clip, advanced fast

    private static readonly string[] FingerNames = { "Index", "Middle", "Ring", "Little" };
    private static readonly int[][] FingerLm = {
        new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 },
        new[] { 13, 14, 15, 16 }, new[] { 17, 18, 19, 20 },
    };
    private static readonly int[] ThumbLm = { 1, 2, 3, 4 };
    private static readonly string[] JointSuffix = { "Proximal", "Intermediate", "Distal" };

    /// <summary>One driven joint: the bone it should move and the segment that drives it.</summary>
    private struct Probe
    {
        public string side;        // "left" / "right"
        public string label;       // e.g. "IndexIntermediate"
        public string boneName;    // HumanBodyBones enum name
        public int[] chain;        // landmark chain [mcp, pip, dip, tip]
        public int jointIndex;     // which joint in the chain is flexed (0..2)
    }

    private class BoneRef
    {
        public HumanBodyBones bone;
        public Transform t;
        public Quaternion rest;
        public bool valid;
    }

    private Animator animator;
    private HandPoseMapper mapper;
    private readonly StringBuilder report = new StringBuilder();
    private readonly List<string> findings = new List<string>();
    private readonly List<BoneRef> boneRefs = new List<BoneRef>();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Install()
    {
        if (FindObjectsByType<BoneAudit>(FindObjectsInactive.Include).Length > 0) return;
        var go = new GameObject("BoneAudit");
        DontDestroyOnLoad(go);
        go.AddComponent<BoneAudit>();
    }

    private void Start()
    {
        mapper = FindObjectsByType<HandPoseMapper>(FindObjectsInactive.Include).FirstOrDefaultCompat();
        if (mapper == null) { Finish("FATAL: no HandPoseMapper in scene."); return; }

        animator = mapper.GetComponent<Animator>();
        if (animator == null) animator = mapper.GetComponentInChildren<Animator>();
        if (animator == null || !animator.isHuman) { Finish("FATAL: no Humanoid Animator."); return; }

        CaptureBones();
        if (boneRefs.Count == 0) { Finish("FATAL: no finger bones resolved."); return; }
        StartCoroutine(RunProbes());
    }

    /// <summary>Snapshot every finger bone's rest rotation before any driving happens.</summary>
    private void CaptureBones()
    {
        foreach (var side in new[] { "left", "right" })
        {
            foreach (var fn in FingerNames)
                foreach (var js in JointSuffix)
                    AddBone(side, fn + js);
            foreach (var js in JointSuffix)
                AddBone(side, "Thumb" + js);
        }
    }

    private void AddBone(string side, string part)
    {
        string prefix = side == "left" ? "Left" : "Right";
        HumanBodyBones bone;
        try { bone = (HumanBodyBones)Enum.Parse(typeof(HumanBodyBones), prefix + part); }
        catch { return; }

        Transform t = null;
        try { t = animator.GetBoneTransform(bone); } catch { }
        if (t == null)
        {
            findings.Add(string.Format("MISSING-BONE {0,-5} {1,-20} animator has no such bone", side, part));
            return;
        }
        boneRefs.Add(new BoneRef { bone = bone, t = t, rest = t.localRotation, valid = true });
    }

    private System.Collections.IEnumerator RunProbes()
    {
        var probes = BuildProbeList();
        report.AppendLine("=== SINGLE-JOINT PROBE MATRIX ===");
        report.AppendLine("Each probe flexes ONE joint by " + PROBE_FLEX_DEG +
            " deg; every other joint is dead straight.");
        report.AppendLine("Correct mapping => exactly the listed bone moves.");
        report.AppendLine();
        report.AppendLine("side   joint                    expected bone     moved (deg)                       verdict");
        report.AppendLine(new string('-', 118));

        foreach (var p in probes)
        {
            // Reset every bone to rest, then apply the one-joint probe.
            ResetPose();
            var clip = BuildProbeClip(p);
            mapper.SetDataset(clip);

            for (int i = 0; i < SETTLE_FRAMES; i++) yield return null;

            var moved = MeasureMoved();
            ReportProbe(p, moved);
        }

        Finish(null);
    }

    private List<Probe> BuildProbeList()
    {
        var list = new List<Probe>();
        foreach (var side in new[] { "left", "right" })
        {
            for (int f = 0; f < FingerNames.Length; f++)
                for (int j = 0; j < 3; j++)
                    list.Add(new Probe
                    {
                        side = side, label = FingerNames[f] + JointSuffix[j],
                        boneName = (side == "left" ? "Left" : "Right") + FingerNames[f] + JointSuffix[j],
                        chain = FingerLm[f], jointIndex = j,
                    });
            for (int j = 0; j < 3; j++)
                list.Add(new Probe
                {
                    side = side, label = "Thumb" + JointSuffix[j],
                    boneName = (side == "left" ? "Left" : "Right") + "Thumb" + JointSuffix[j],
                    chain = ThumbLm, jointIndex = j,
                });
        }
        return list;
    }

    /// <summary>
    /// Synthetic hand, dead straight, with exactly one joint flexed.
    ///
    /// Geometry is built directly in MediaPipe image space (x right, y DOWN,
    /// z toward viewer) because that is what the mapper consumes; the mapper's
    /// own coordinate conversion then maps it into the rig. Fingers are laid
    /// out with real anatomical spread and knuckle curl so the palm basis
    /// (wrist -> index MCP -> middle MCP) is well conditioned -- the earlier
    /// collinear synthetic hand produced a near-degenerate basis and therefore
    /// meaningless measurements.
    /// </summary>
    private LandmarkDataset BuildProbeClip(Probe p)
    {
        var ds = new LandmarkDataset { fps = PROBE_HOLD_FPS, frame_count = 1, frames = new List<FrameData>() };
        var frame = new FrameData { frame_index = 0, hands = new List<HandData>() };

        foreach (var sideName in new[] { "Left", "Right" })
        {
            var pts = BuildStraightHand(sideName == "Left");
            // Flex the target joint on the probe's side only.
            if (sideName.ToLowerInvariant() == p.side)
                FlexJoint(pts, p.chain, p.jointIndex, PROBE_FLEX_DEG, sideName == "Left");

            frame.hands.Add(new HandData
            {
                hand_index = 0,
                handedness = sideName,
                handedness_score = 1f,
                landmarks = pts,
            });
        }
        ds.frames.Add(frame);
        return ds;
    }

    /// <summary>
    /// A relaxed, anatomically plausible open hand in MediaPipe image space.
    /// Wrist low, knuckles in an arc above it, fingers fanned, slight natural
    /// curl, thumb abducted to its own side. Index points are laid out in
    /// landmark order: wrist, thumb(4), index(4), middle(4), ring(4), little(4).
    /// </summary>
    private static List<LandmarkPoint> BuildStraightHand(bool left)
    {
        var p = new List<LandmarkPoint>();
        float sideSign = left ? -1f : 1f;   // thumb abducts toward its own side

        p.Add(new LandmarkPoint { x = 0.500f, y = 0.700f, z = 0.000f });   // 0 wrist

        // Thumb: abducted out to the side, angled ~35 deg up from the wrist.
        const float thumbPitch = 0.62f;      // radians-ish rise per segment
        for (int k = 0; k < 4; k++)
        {
            float t = k + 1;
            p.Add(new LandmarkPoint
            {
                x = 0.500f + sideSign * 0.055f * t * 0.72f,
                y = 0.700f - 0.050f * t * thumbPitch / 0.62f,
                z = -0.010f * k,
            });
        }

        // Four fingers: MCP arc across the palm, then straight-ish segments.
        // Per-finger MCP x, MCP y, length scale, fan (radians of outward lean).
        float[] mcpX = { 0.545f, 0.512f, 0.479f, 0.448f };
        float[] mcpY = { 0.590f, 0.585f, 0.592f, 0.602f };
        float[] len = { 1.00f, 1.06f, 0.96f, 0.80f };
        float[] fan = { -0.10f, 0.00f, 0.10f, 0.20f };

        for (int f = 0; f < 4; f++)
        {
            float baseX = mcpX[f];
            float baseY = mcpY[f];
            // Four landmarks per finger: MCP (k=0) then PIP, DIP, TIP. MediaPipe
            // hands carry four points per finger, so emitting three yields a
            // 17-point hand and every later landmark index overruns the list.
            float segLen = 0.040f * len[f];
            for (int k = 0; k < 4; k++)
            {
                p.Add(new LandmarkPoint
                {
                    x = baseX + fan[f] * k * 0.030f,
                    y = baseY - segLen * k,
                    z = -0.006f * f,
                });
            }
        }
        if (p.Count != 21)
            throw new Exception("BoneAudit: synthetic hand must have 21 landmarks, got " + p.Count);
        return p;
    }

    /// <summary>
    /// Rotate the sub-chain distal to joint j about joint j's own landmark,
    /// in the palm plane (rotation about the image z axis). Because MediaPipe
    /// y points DOWN, a positive angle here bends the finger toward the palm.
    /// Only the driven segment's direction changes; every other joint stays
    /// exactly straight.
    /// </summary>
    private static void FlexJoint(List<LandmarkPoint> pts, int[] chain, int jointIndex,
                                   float degrees, bool left)
    {
        int pivot = LandmarkIndex(chain, jointIndex);
        LandmarkPoint o = pts[pivot];
        float rad = degrees * Mathf.Deg2Rad * (left ? -1f : 1f);
        float c = Mathf.Cos(rad), s = Mathf.Sin(rad);

        // Rotate every point distal to the pivot, in image space, about z.
        for (int i = pivot + 1; i < pts.Count; i++)
        {
            if (!IsDistalTo(pts, chain, pivot, i)) continue;
            float dx = pts[i].x - o.x;
            float dy = pts[i].y - o.y;
            pts[i].x = o.x + dx * c - dy * s;
            pts[i].y = o.y + dx * s + dy * c;
        }
    }

    /// <summary>Map chain-local joint index (0..2) to the global landmark index.</summary>
    private static int LandmarkIndex(int[] chain, int jointIndex)
    {
        return chain[Mathf.Clamp(jointIndex, 0, chain.Length - 1)];
    }

    /// <summary>
    /// True when global index i lies distal to global index pivot along the
    /// same finger chain. Rotating those (and only those) flexes one joint.
    /// </summary>
    private static bool IsDistalTo(List<LandmarkPoint> pts, int[] chain, int pivot, int i)
    {
        int chainPos = -1, pivotPos = -1;
        for (int k = 0; k < chain.Length; k++)
        {
            if (chain[k] == i) chainPos = k;
            if (chain[k] == pivot) pivotPos = k;
        }
        // Not part of this chain, or before the pivot: leave alone.
        if (chainPos < 0 || pivotPos < 0 || chainPos <= pivotPos) return false;
        // Must be in the same contiguous run of landmarks after pivot.
        for (int k = pivotPos; k < chainPos; k++)
            if (k + 1 < chain.Length && chain[k + 1] != chain[k] + 1) return false;
        return true;
    }

    private void ResetPose()
    {
        foreach (var b in boneRefs)
            if (b.valid && b.t != null) b.t.localRotation = b.rest;
    }

    /// <summary>
    /// Rotation magnitude of every tracked bone relative to its captured rest,
    /// in degrees. Uses a quaternion delta, so a bone with no child transform
    /// (tipless Distal) measures correctly -- unlike the earlier version.
    /// </summary>
    private List<KeyValuePair<string, float>> MeasureMoved()
    {
        var moved = new List<KeyValuePair<string, float>>();
        foreach (var b in boneRefs)
        {
            float deg = 0f;
            if (b.valid && b.t != null)
            {
                Quaternion delta = b.t.localRotation * Quaternion.Inverse(b.rest);
                float c = Mathf.Clamp(Mathf.Abs(delta.w), -1f, 1f);
                deg = 2f * Mathf.Acos(c) * Mathf.Rad2Deg;
            }
            moved.Add(new KeyValuePair<string, float>(b.bone.ToString(), deg));
        }
        return moved;
    }

    private void ReportProbe(Probe p, List<KeyValuePair<string, float>> moved)
    {
        var above = new List<string>();
        foreach (var kv in moved)
            if (kv.Value > MOVED_EPS_DEG) above.Add(kv.Key + "=" + kv.Value.ToString("0.0"));

        bool expectedMoved = false;
        foreach (var s in above) if (s.StartsWith(p.boneName + "=")) expectedMoved = true;

        string verdict;
        if (above.Count == 0)
        {
            verdict = "DEAD: nothing moved -> bone never driven";
            findings.Add(string.Format("DEAD-JOINT   {0,-5} {1,-20} probe moved no bone at all",
                p.side, p.label));
        }
        else if (!expectedMoved)
        {
            verdict = "MIS-MAPPED: expected bone did NOT move";
            findings.Add(string.Format("MIS-MAPPED   {0,-5} {1,-20} expected {2} to move; moved: {3}",
                p.side, p.label, p.boneName, string.Join(", ", above.ToArray())));
        }
        else if (above.Count == 1)
        {
            verdict = "ok (isolated)";
        }
        else
        {
            // More than one bone moved: acceptable only if the extras are tiny.
            verdict = above.Count + " bones moved (check for bleed)";
        }

        report.AppendLine(string.Format("{0,-6} {1,-24} {2,-17} {3,-33} {4}",
            p.side, p.label, p.boneName,
            above.Count == 0 ? "(none)" : string.Join(", ", above.ToArray()),
            verdict));
    }

    private void Finish(string fatal)
    {
        report.AppendLine();
        report.AppendLine("=== FINDINGS ===");
        if (fatal != null) report.AppendLine(fatal);
        if (findings.Count == 0) report.AppendLine("none - every joint drove exactly its own bone");
        else foreach (var f in findings) report.AppendLine(f);

        string text = report.ToString();
        Debug.Log("[BoneAudit]\n" + text);
        TryWrite(text);
    }

    private void TryWrite(string text)
    {
        string[] tries = {
            Path.Combine(Application.dataPath, "..", "bone_audit_report.txt"),
            Path.Combine(Application.persistentDataPath, "bone_audit_report.txt"),
        };
        foreach (var p in tries)
        {
            try { File.WriteAllText(p, text); Debug.Log("[BoneAudit] report -> " + p); return; }
            catch { }
        }
        Debug.LogWarning("[BoneAudit] could not write report file; see Console.");
    }
}

internal static class BoneAuditExtensions
{
    public static T FirstOrDefaultCompat<T>(this T[] arr) where T : class
    {
        return arr != null && arr.Length > 0 ? arr[0] : null;
    }
}