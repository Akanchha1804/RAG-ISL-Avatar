using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

/// <summary>
/// Runtime verification of the landmark -> bone mapping, driven by the
/// mapper's own recorded drive inputs rather than by inferred motion.
///
/// Why this replaced the earlier aggregate probe: measuring which bones moved
/// conflates the mapping with the anatomical clamp, the smoothing transient
/// and the parent-chain pose, so a "bone did not move" result is ambiguous.
/// HandPoseMapper now optionally records, per bone, the exact segment
/// direction it was driven by plus the pre-clamp flexion angle and whether the
/// clamp saturated. Those are the mapper's actual inputs, so they localise a
/// fault precisely.
///
/// For each of the 30 driven joints the probe:
///   1. identifies which landmark pair actually drove each bone, by matching
///      the recorded segment direction against every candidate pair;
///   2. compares the pre-clamp flexion angle against a no-flexion baseline;
///   3. reports whether the clamp saturated (a bone pinned at its limit cannot
///      respond, which is a real, separate defect from a wrong mapping).
///
/// Arms and the Animator are disabled so the only writer under test is the
/// mapper's retargeting.
/// </summary>
public class FingerRetargetProbe
{
    private const float PROBE_FLEX_DEG = 40f;
    private const int SETTLE_FRAMES = 150;
    private const float ANGLE_EPS_DEG = 3f;

    private static readonly string[] FingerNames = { "Index", "Middle", "Ring", "Little" };
    private static readonly int[][] FingerLm = {
        new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 },
        new[] { 13, 14, 15, 16 }, new[] { 17, 18, 19, 20 },
    };
    private static readonly int[] ThumbLm = { 1, 2, 3, 4 };
    private static readonly string[] JointSuffix = { "Proximal", "Intermediate", "Distal" };

    private struct Probe
    {
        public string side, label, boneName;
        public int[] chain;
        public int jointIndex;
        public int segA, segB;      // landmark pair expected to drive this bone
    }

    private Animator _animator;
    private HandPoseMapper _mapper;
    private readonly StringBuilder _report = new StringBuilder();
    private readonly List<string> _findings = new List<string>();
    private readonly Dictionary<HumanBodyBones, float> _baselineFlex =
        new Dictionary<HumanBodyBones, float>();
    private bool _armIkWasEnabled, _animatorWasEnabled;

    [UnityTest]
    public IEnumerator EachJointDrivesItsOwnSegment()
    {
        yield return LoadSceneWithAvatar();
        if (_mapper == null || _animator == null)
        {
            Assert.Fail("Avatar/HandPoseMapper not found in scene; probe cannot run.");
            yield break;
        }

        _armIkWasEnabled = _mapper.enableArmIK;
        _animatorWasEnabled = _animator.enabled;
        _mapper.enableArmIK = false;
        _animator.enabled = false;
        _mapper.CaptureDiagnostics = true;

        // Headless/offline the scene's AvatarWebSocketClient logs a connection
        // error that the test framework would otherwise treat as a failure
        // (and abort the probe) before any measurement happens.
        LogAssert.ignoreFailingMessages = true;

        // ---- baseline: no flexion anywhere ----
        var baseClip = BuildClip(null);
        _mapper.SetDataset(baseClip);
        yield return Settle();
        _baselineFlex.Clear();
        foreach (var kv in _mapper.LastDiagnostics)
            _baselineFlex[kv.Key] = kv.Value.flexAngPreClamp;

        _report.AppendLine("=== SEGMENT -> BONE VERIFICATION (mapper-recorded landmark indices) ===");
        _report.AppendLine("The mapper records the landmark pair it drove each bone with, so the mapping");
        _report.AppendLine("is asserted directly. flexAng = pre-clamp angle from rest (deg).");
        _report.AppendLine();
        _report.AppendLine("side   joint                    segment  flexAng base->probe  delta   clamp  damp   verdict");
        _report.AppendLine(new string('-', 104));

        foreach (var p in BuildProbes())
        {
            _mapper.SetDataset(BuildClip(p));
            yield return Settle();
            ReportProbe(p);
        }

        _report.AppendLine();
        _report.AppendLine("=== FINDINGS ===");
        if (_findings.Count == 0) _report.AppendLine("none - every bone is driven by its own landmark segment");
        else foreach (var f in _findings) _report.AppendLine(f);

        Write(_report.ToString());
        Debug.Log("[FingerRetargetProbe]\n" + _report);

        _mapper.CaptureDiagnostics = false;
        _mapper.enableArmIK = _armIkWasEnabled;
        _animator.enabled = _animatorWasEnabled;

        Assert.IsEmpty(_findings, "Segment->bone verification failed (see report).");
    }

    private IEnumerator LoadSceneWithAvatar()
    {
        var scene = SceneManager.GetSceneByName("SampleScene");
        if (!scene.isLoaded)
        {
            SceneManager.LoadScene("SampleScene", LoadSceneMode.Single);
            // The load is asynchronous and, headless, takes far more than the
            // two frames this used to yield for, so the probe reported
            // "not found in scene" whenever it ran without a scene already
            // open (batch mode / CI). Wait for the mapper to actually exist.
            float deadline = Time.realtimeSinceStartup + 30f;
            while (FindFirstMapper() == null && Time.realtimeSinceStartup < deadline)
                yield return null;
        }
        _mapper = FindFirstMapper();
        if (_mapper == null) yield break;
        _animator = _mapper.GetComponent<Animator>();
        if (_animator == null) _animator = _mapper.GetComponentInChildren<Animator>();
    }

    private static HandPoseMapper FindFirstMapper()
    {
        foreach (var m in UnityEngine.Object.FindObjectsByType<HandPoseMapper>(
            FindObjectsInactive.Include))
            return m;
        return null;
    }

    private List<Probe> BuildProbes()
    {
        var list = new List<Probe>();
        foreach (var side in new[] { "left", "right" })
        {
            for (int f = 0; f < FingerNames.Length; f++)
                for (int j = 0; j < 3; j++)
                    list.Add(new Probe
                    {
                        side = side,
                        label = FingerNames[f] + JointSuffix[j],
                        boneName = (side == "left" ? "Left" : "Right") + FingerNames[f] + JointSuffix[j],
                        chain = FingerLm[f],
                        jointIndex = j,
                        segA = FingerLm[f][j], segB = FingerLm[f][j + 1],
                    });
            for (int j = 0; j < 3; j++)
                list.Add(new Probe
                {
                    side = side,
                    label = "Thumb" + JointSuffix[j],
                    boneName = (side == "left" ? "Left" : "Right") + "Thumb" + JointSuffix[j],
                    chain = ThumbLm,
                    jointIndex = j,
                    segA = ThumbLm[j], segB = ThumbLm[j + 1],
                });
        }
        return list;
    }

    private void ReportProbe(Probe p)
    {
        HumanBodyBones hb;
        try { hb = (HumanBodyBones)Enum.Parse(typeof(HumanBodyBones), p.boneName); }
        catch { _findings.Add("BAD-BONE-NAME " + p.boneName); return; }

        if (!_mapper.LastDiagnostics.TryGetValue(hb, out var d))
        {
            _findings.Add("NO-DIAG " + p.boneName + " : mapper never drove this bone");
            return;
        }

        // The mapper recorded which landmark pair it actually used, so the
        // segment -> bone mapping is asserted directly with no geometric
        // inference. This is the part that cannot be faked by test geometry.
        bool correctSegment = d.startIdx == p.segA && d.endIdx == p.segB;
        float baseAng = _baselineFlex.ContainsKey(hb) ? _baselineFlex[hb] : 0f;
        float delta = d.flexAngPreClamp - baseAng;

        string verdict;
        if (!correctSegment)
        {
            verdict = "MIS-MAPPED";
            _findings.Add("MIS-MAPPED   " + p.side + " " + p.label + " : " + p.boneName
                + " driven by (" + d.startIdx + "," + d.endIdx + "), expected ("
                + p.segA + "," + p.segB + ")");
        }
        else if (Mathf.Abs(delta) < ANGLE_EPS_DEG)
        {
            verdict = d.clamped ? "SATURATED at clamp" : "NO-RESPONSE";
            _findings.Add("NO-RESPONSE " + p.side + " " + p.label + " : " + p.boneName
                + " correctly driven by (" + p.segA + "," + p.segB + ") but flexAng moved only "
                + delta.ToString("0.0") + " deg" + (d.clamped
                    ? "; CLAMPED (limit " + d.maxFlexDeg + ", rest-offset " + baseAng.ToString("0") + ")"
                    : ""));
        }
        else verdict = "ok";

        _report.AppendLine(string.Format("{0,-6} {1,-24} ({2},{3}){4,-4} {5,-16} {6,-8} {7,-9} {8}",
            p.side, p.label, p.segA, p.segB,
            correctSegment ? "" : "!" + d.startIdx + "," + d.endIdx,
            baseAng.ToString("0") + "->" + d.flexAngPreClamp.ToString("0"),
            "d=" + delta.ToString("+0;-0"),
            d.clamped ? "CLAMP" : "-",
            d.damp.ToString("0.00"), verdict));
    }

    private IEnumerator Settle()
    {
        var prev = new Dictionary<HumanBodyBones, float>();
        for (int f = 0; f < SETTLE_FRAMES; f++)
        {
            var now = new Dictionary<HumanBodyBones, float>();
            foreach (var kv in _mapper.LastDiagnostics) now[kv.Key] = kv.Value.flexAngPreClamp;
            if (f > 4)
            {
                float maxDelta = 0f;
                foreach (var kv in now)
                {
                    float p = prev.ContainsKey(kv.Key) ? prev[kv.Key] : 0f;
                    maxDelta = Mathf.Max(maxDelta, Mathf.Abs(kv.Value - p));
                }
                if (maxDelta < 0.05f) yield break;
            }
            prev.Clear();
            foreach (var kv in now) prev[kv.Key] = kv.Value;
            yield return null;
        }
    }

    private LandmarkDataset BuildClip(Probe? p)
    {
        var ds = new LandmarkDataset { fps = 60f, frame_count = 1, frames = new List<FrameData>() };
        var frame = new FrameData { frame_index = 0, hands = new List<HandData>() };
        foreach (var sideName in new[] { "Left", "Right" })
        {
            var pts = BuildStarHand(sideName == "Left");
            if (p.HasValue && sideName.ToLowerInvariant() == p.Value.side)
                FlexJoint(pts, p.Value.chain, p.Value.jointIndex, PROBE_FLEX_DEG, sideName == "Left");
            frame.hands.Add(new HandData { hand_index = 0, handedness = sideName, handedness_score = 1f, landmarks = pts });
        }
        ds.frames.Add(frame);
        return ds;
    }

    /// <summary>Star-laid hand: every phalanx collinear with wrist->knuckle, so true curl is 0.</summary>
    private static List<LandmarkPoint> BuildStarHand(bool left)
    {
        var wrist = new Vector3(0.50f, 0.70f, 0f);
        float s = left ? -1f : 1f;
        var dir = new[]
        {
            new Vector3(0.55f * s, 0.84f, 0f),
            new Vector3(0.34f, 0.94f, 0f),
            new Vector3(0.11f, 0.99f, 0f),
            new Vector3(-0.13f, 0.99f, 0f),
            new Vector3(-0.38f, 0.93f, 0f),
        };
        for (int i = 0; i < dir.Length; i++) dir[i] = dir[i].normalized;
        float[][] radii = {
            new[] { 0.035f, 0.060f, 0.082f, 0.100f },
            new[] { 0.090f, 0.130f, 0.160f, 0.180f },
            new[] { 0.092f, 0.136f, 0.168f, 0.188f },
            new[] { 0.088f, 0.130f, 0.158f, 0.176f },
            new[] { 0.078f, 0.112f, 0.136f, 0.152f },
        };
        var p = new List<LandmarkPoint>();
        p.Add(new LandmarkPoint { x = wrist.x, y = wrist.y, z = wrist.z });
        for (int c = 0; c < 5; c++)
        {
            float z = -0.004f * c;
            for (int k = 0; k < 4; k++)
            {
                Vector3 q = wrist + dir[c] * radii[c][k] + new Vector3(0f, 0f, z);
                p.Add(new LandmarkPoint { x = q.x, y = q.y, z = q.z });
            }
        }
        if (p.Count != 21) throw new Exception("synthetic hand must have 21 landmarks, got " + p.Count);
        return p;
    }

    private static void FlexJoint(List<LandmarkPoint> pts, int[] chain, int jointIndex, float degrees, bool left)
    {
        int pivot = chain[Mathf.Clamp(jointIndex, 0, chain.Length - 1)];
        LandmarkPoint o = pts[pivot];
        float rad = degrees * Mathf.Deg2Rad * (left ? -1f : 1f);
        float c = Mathf.Cos(rad), s = Mathf.Sin(rad);
        for (int i = pivot + 1; i < pts.Count; i++)
        {
            if (!IsDistalTo(chain, pivot, i)) continue;
            float dx = pts[i].x - o.x, dy = pts[i].y - o.y;
            pts[i].x = o.x + dx * c - dy * s;
            pts[i].y = o.y + dx * s + dy * c;
        }
    }

    private static bool IsDistalTo(int[] chain, int pivot, int i)
    {
        int chainPos = -1, pivotPos = -1;
        for (int k = 0; k < chain.Length; k++)
        {
            if (chain[k] == i) chainPos = k;
            if (chain[k] == pivot) pivotPos = k;
        }
        if (chainPos < 0 || pivotPos < 0 || chainPos <= pivotPos) return false;
        for (int k = pivotPos; k < chainPos; k++)
            if (k + 1 < chain.Length && chain[k + 1] != chain[k] + 1) return false;
        return true;
    }

    private static void Write(string text)
    {
        string[] tries = {
            Path.Combine(Application.dataPath, "..", "bone_audit_report.txt"),
            Path.Combine(Application.persistentDataPath, "bone_audit_report.txt"),
        };
        foreach (var p in tries)
        {
            try { File.WriteAllText(p, text); Debug.Log("[FingerRetargetProbe] report -> " + p); return; }
            catch { }
        }
    }
}