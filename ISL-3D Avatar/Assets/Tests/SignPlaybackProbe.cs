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
/// Measures whether the avatar actually performs a given sign, using the
/// real landmark clip rather than synthetic data.
///
/// "The avatar isn't doing namaste correctly" is a claim about the rendered
/// pose, not about the mapping table. This test loads the real
/// gloss_namaste.json clip, lets the mapper drive the rig, then reports the
/// quantities that define the gesture:
///
///   * wrist separation - namaste brings the hands together, so this must
///                        shrink well below the rest-pose span;
///   * wrist height     - the gesture sits in front of the chest, not at the
///                        hips;
///   * palm facing      - palms face each other, so the two palm normals
///                        should be roughly anti-parallel;
///   * wrist rotation   - the wrist must actually turn.
/// </summary>
public class SignPlaybackProbe
{
    private const string CLIP = "gloss_namaste.json";

    private Animator _animator;
    private HandPoseMapper _mapper;
    private readonly StringBuilder _report = new StringBuilder();
    private readonly List<string> _findings = new List<string>();
    // Hand local rotations captured at the true rest pose, so the wrist
    // rotation check compares against rest instead of against itself.
    private Quaternion _restLocalL = Quaternion.identity;
    private Quaternion _restLocalR = Quaternion.identity;

    [UnityTest]
    public IEnumerator NamasteBringsHandsTogether()
    {
        yield return LoadSceneWithAvatar();
        if (_mapper == null || _animator == null)
        {
            Assert.Fail("Avatar/HandPoseMapper not found in scene.");
            yield break;
        }

        // Only the mapper may write bone rotations.
        bool animWas = _animator.enabled;
        _animator.enabled = false;
        _mapper.CaptureDiagnostics = true;

        // This probe is about geometry, and it runs headless/offline (batch
        // mode, no websocket server), where the scene's AvatarWebSocketClient
        // logs a connection error that would otherwise fail the test as an
        // unhandled log message before a single frame is measured.
        LogAssert.ignoreFailingMessages = true;

        // ---- rest baseline ----
        // SetDataset(empty) is REJECTED by the mapper, so it cannot be used
        // to establish a rest pose (that was the old, silent bug: "rest" was
        // measured with the previous clip still playing). Stop playback and
        // snap the rig to rest instead, and remember the rest hand rotations
        // so the wrist-rotation check has something real to compare against.
        _mapper.playAnimation = false;
        _mapper.ResetPose();
        yield return new WaitForSeconds(0.3f);
        if (Bone(HumanBodyBones.LeftHand) != null)
            _restLocalL = Bone(HumanBodyBones.LeftHand).localRotation;
        if (Bone(HumanBodyBones.RightHand) != null)
            _restLocalR = Bone(HumanBodyBones.RightHand).localRotation;
        var rest = Measure();

        // ---- real namaste clip ----
        // HandPoseMapper.Start already auto-loads landmarks.json, so simply
        // waiting for "a dataset" returns that clip immediately and the test
        // silently measured the WRONG clip (232 frames, not namaste). Wait
        // for a DIFFERENT dataset object (or one named namaste).
        _mapper.playAnimation = true;
        var before = _mapper.GetDataset();
        _mapper.LoadLandmarkFile(CLIP);
        float waited = 0f;
        while (waited < 20f)
        {
            var cur = _mapper.GetDataset();
            bool isNew = cur != null && !ReferenceEquals(cur, before)
                && cur.frames != null && cur.frames.Count > 0;
            bool isNamaste = cur != null && cur.video_name != null
                && cur.video_name.ToLowerInvariant().Contains("namaste");
            if (isNew || isNamaste)
                break;
            waited += Time.deltaTime;
            yield return null;
        }

        var ds = _mapper.GetDataset();
        if (ds == null || ds.frames == null || ds.frames.Count == 0)
        {
            _report.AppendLine("FAIL: clip " + CLIP + " never loaded (backend down or file missing).");
            Write(_report.ToString());
            Assert.Fail("Clip did not load.");
            yield break;
        }

        // Sample while the clip plays.
        var samples = new List<Dictionary<string, float>>();
        // Worst palm-vs-chest penetration over the clip (metres) and how many
        // palm vertices the torso was drawn over at that moment.
        float penL = 0f, penR = 0f, penAtL = 0f, penAtR = 0f;
        int penInL = 0, penInR = 0;
        for (int i = 0; i < 90; i++)
        {
            yield return new WaitForSeconds(0.05f);
            samples.Add(Measure());
            int inL, inR; float dL, dR;
            _mapper.MeasureHandPenetration(true, out inL, out dL);
            _mapper.MeasureHandPenetration(false, out inR, out dR);
            if (dL > penL) { penL = dL; penInL = inL; penAtL = Bone(HumanBodyBones.LeftHand).position.y; }
            if (dR > penR) { penR = dR; penInR = inR; penAtR = Bone(HumanBodyBones.RightHand).position.y; }
        }
        _report.AppendLine();
        _report.AppendLine("=== CHEST PENETRATION (worst palm vertex over the clip) ===");
        _report.AppendLine("left  deepest " + penL.ToString("0.0000") + " m, vertices inside "
            + penInL + " at wrist y " + penAtL.ToString("0.000"));
        _report.AppendLine("right deepest " + penR.ToString("0.0000") + " m, vertices inside "
            + penInR + " at wrist y " + penAtR.ToString("0.000"));
        // A millimetre of grazing is measurement noise; a centimetre is the
        // palm being drawn over by the torso, which is the reported defect.
        if (penL > 0.01f)
            _findings.Add("CHEST-SINK left: palm " + penL.ToString("0.0000")
                + " m inside the torso (" + penInL + " vertices covered).");
        if (penR > 0.01f)
            _findings.Add("CHEST-SINK right: palm " + penR.ToString("0.0000")
                + " m inside the torso (" + penInR + " vertices covered).");

        // Hold the final frame for a long time to see if smoothing converges.
        _mapper.SetFrameIndex(ds.frames.Count - 1);
        for (int i = 0; i < 120; i++) yield return null;
        var settled = Measure();
        _report.AppendLine();
        _report.AppendLine("=== AFTER LONG SETTLE (final frame held 120 frames) ===");
        _report.AppendLine("L wrist: " + Bone(HumanBodyBones.LeftHand).position.ToString("F3")
            + "  target: " + _mapper.LastTargetLeft.ToString("F3")
            + "  err: " + Vector3.Distance(Bone(HumanBodyBones.LeftHand).position, _mapper.LastTargetLeft).ToString("0.000"));
        _report.AppendLine("R wrist: " + Bone(HumanBodyBones.RightHand).position.ToString("F3")
            + "  target: " + _mapper.LastTargetRight.ToString("F3")
            + "  err: " + Vector3.Distance(Bone(HumanBodyBones.RightHand).position, _mapper.LastTargetRight).ToString("0.000"));

        // Per-frame wrist x positions, to see the actual trajectory.
        _report.AppendLine();
        _report.AppendLine("=== WRIST TRAJECTORY (every 6th sample) ===");
        _report.AppendLine("sample   L.x      R.x      sep     L.y     R.y     TgtL.x  TgtR.x  ErrL    ErrR");
        for (int i = 0; i < samples.Count; i += 6)
        {
            var s = samples[i];
            float tl = _mapper.LastTargetLeft.x, tr = _mapper.LastTargetRight.x;
            float el = Mathf.Abs(s["lx"] - tl), er = Mathf.Abs(s["rx"] - tr);
            _report.AppendLine(string.Format("{0,-8} {1,-8:0.000} {2,-8:0.000} {3,-7:0.000} {4,-6:0.000} {5,-6:0.000} {6,-7:0.000} {7,-7:0.000} {8,-6:0.000} {9,-6:0.000}",
                i, s["lx"], s["rx"], s["sep"], s["ly"], s["ry"], tl, tr, el, er));
        }
        _report.AppendLine("rest L.x=" + rest["lx"].ToString("0.000") + " R.x=" + rest["rx"].ToString("0.000"));

        // 3D distance from wrist to target, to see if it's x, y or z.
        _report.AppendLine();
        _report.AppendLine("=== 3D WRIST-TO-TARGET ERROR ===");
        var lw3 = Bone(HumanBodyBones.LeftHand);
        var rw3 = Bone(HumanBodyBones.RightHand);
        if (lw3 != null && rw3 != null)
        {
            float dl = Vector3.Distance(lw3.position, _mapper.LastTargetLeft);
            float dr = Vector3.Distance(rw3.position, _mapper.LastTargetRight);
            _report.AppendLine("left  wrist->target: " + dl.ToString("0.000") + " m");
            _report.AppendLine("right wrist->target: " + dr.ToString("0.000") + " m");
            _report.AppendLine("left  wrist: " + lw3.position.ToString("F3") + "  target: " + _mapper.LastTargetLeft.ToString("F3"));
            _report.AppendLine("right wrist: " + rw3.position.ToString("F3") + "  target: " + _mapper.LastTargetRight.ToString("F3"));

            // Decompose the target height to see which term is off.
            var chest = Bone(HumanBodyBones.Chest) ?? Bone(HumanBodyBones.Spine);
            var neck = Bone(HumanBodyBones.Neck) ?? Bone(HumanBodyBones.Head);
            _report.AppendLine("chest pos: " + (chest != null ? chest.position.ToString("F3") : "null"));
            _report.AppendLine("neck  pos: " + (neck != null ? neck.position.ToString("F3") : "null"));
            _report.AppendLine("hips  pos: " + (Bone(HumanBodyBones.Hips) != null ? Bone(HumanBodyBones.Hips).position.ToString("F3") : "null"));

            // Shoulder positions (rest), to compare against the targets.
            var ls = Bone(HumanBodyBones.LeftUpperArm);
            var rs = Bone(HumanBodyBones.RightUpperArm);
            _report.AppendLine("L shoulder: " + (ls != null ? ls.position.ToString("F3") : "null"));
            _report.AppendLine("R shoulder: " + (rs != null ? rs.position.ToString("F3") : "null"));

            // Arm lengths, to see if the target is reachable at all.
            var le = Bone(HumanBodyBones.LeftLowerArm);
            var re = Bone(HumanBodyBones.RightLowerArm);
            if (ls != null && le != null && lw3 != null)
                _report.AppendLine("L arm len: " + (ls.position - le.position).magnitude.ToString("F3")
                    + " + " + (le.position - lw3.position).magnitude.ToString("F3")
                    + " = " + (ls.position - lw3.position).magnitude.ToString("F3")
                    + " | dist to target: " + (ls.position - _mapper.LastTargetLeft).magnitude.ToString("F3"));
            if (rs != null && re != null && rw3 != null)
                _report.AppendLine("R arm len: " + (rs.position - re.position).magnitude.ToString("F3")
                    + " + " + (re.position - rw3.position).magnitude.ToString("F3")
                    + " = " + (rs.position - rw3.position).magnitude.ToString("F3")
                    + " | dist to target: " + (rs.position - _mapper.LastTargetRight).magnitude.ToString("F3"));

            // Are the arm bones actually rotating? Output their world directions.
            if (ls != null && le != null)
            {
                Vector3 uDir = (le.position - ls.position).normalized;
                Vector3 lDir = (lw3.position - le.position).normalized;
                _report.AppendLine("L upper dir: " + uDir.ToString("F3") + " (angle from down: "
                    + Vector3.Angle(uDir, Vector3.down).ToString("0") + " deg)");
                _report.AppendLine("L lower dir: " + lDir.ToString("F3") + " (elbow bend: "
                    + Vector3.Angle(uDir, lDir).ToString("0") + " deg)");
            }
            if (rs != null && re != null)
            {
                Vector3 uDir = (re.position - rs.position).normalized;
                Vector3 lDir = (rw3.position - re.position).normalized;
                _report.AppendLine("R upper dir: " + uDir.ToString("F3") + " (angle from down: "
                    + Vector3.Angle(uDir, Vector3.down).ToString("0") + " deg)");
                _report.AppendLine("R lower dir: " + lDir.ToString("F3") + " (elbow bend: "
                    + Vector3.Angle(uDir, lDir).ToString("0") + " deg)");
            }

            // Wrist rotation: is the Hand bone actually turning?
            _report.AppendLine();
            _report.AppendLine("=== WRIST ROTATION ===");
            _report.AppendLine("L hand in boneMap: " + _mapper.HasBone(HumanBodyBones.LeftHand));
            _report.AppendLine("R hand in boneMap: " + _mapper.HasBone(HumanBodyBones.RightHand));
            _report.AppendLine("L hand localRot (rest): " + _restLocalL.eulerAngles.ToString("F1"));
            _report.AppendLine("R hand localRot (rest): " + _restLocalR.eulerAngles.ToString("F1"));
            _report.AppendLine("L hand localRot (clip): " + lw3.localRotation.eulerAngles.ToString("F1"));
            _report.AppendLine("R hand localRot (clip): " + rw3.localRotation.eulerAngles.ToString("F1"));
            // Compare against the captured REST rotation (the old code
            // compared the transform against itself, so this always read 0).
            float lwr = Quaternion.Angle(_restLocalL, lw3.localRotation);
            float rwr = Quaternion.Angle(_restLocalR, rw3.localRotation);
            _report.AppendLine("L wrist rotation: " + lwr.ToString("0.0") + " deg");
            _report.AppendLine("R wrist rotation: " + rwr.ToString("0.0") + " deg");
            if (lwr < 5f && rwr < 5f)
                _findings.Add("WRIST-STILL: neither wrist rotated vs rest (L " + lwr.ToString("0.0")
                    + " deg, R " + rwr.ToString("0.0") + " deg) - the hand bone is not being driven.");

            // Palm basis: what the mapper computed for avatar vs landmark.
            var diag = _mapper.LastDiagnostics;
            if (diag != null && diag.Count > 0)
            {
                float maxFlex = 0f;
                foreach (var kv in diag)
                    if (kv.Value.flexAngPreClamp > maxFlex) maxFlex = kv.Value.flexAngPreClamp;
                _report.AppendLine("max finger flexAng in clip: " + maxFlex.ToString("0.0") + " deg");
            }

            // Raw landmark palm normal vs the avatar's configured rest
            // normal, per side. A chirality/mirror bug shows up as a dot
            // near -1 (frame faces opposite ways). The old code computed
            // this from the POSED bones, which hid the mismatch.
            var ds2 = _mapper.GetDataset();
            if (ds2 != null && ds2.frames != null && ds2.frames.Count > 0)
            {
                var frame = ds2.frames[0];
                foreach (var h in frame.hands)
                {
                    if (h == null || h.landmarks == null || h.landmarks.Count != 21) continue;
                    bool isL = h.handedness != null && h.handedness.ToLower().Contains("left");
                    var handT = Bone(isL ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand);
                    if (handT == null) continue;
                    Vector3 raw = _mapper.LandmarkPalmNormal(h);
                    Vector3 av = isL ? _mapper.LeftPalmNormal : _mapper.RightPalmNormal;
                    _report.AppendLine((isL ? "L" : "R") + " raw landmark palm normal: " + raw.ToString("F3")
                        + "  avatar rest: " + av.ToString("F3")
                        + "  dot: " + Vector3.Dot(raw, av).ToString("0.00"));
                }
            }
            _report.AppendLine("armIK enabled: " + _mapper.IsArmIkEnabled);
        }

        _report.AppendLine("=== SIGN PLAYBACK: " + CLIP + " (" + ds.frames.Count + " frames) ===");
        _report.AppendLine();
        _report.AppendLine("quantity                rest      clip(min..max)       verdict");
        _report.AppendLine(new string('-', 80));

        Add("wrist separation (m)", "sep", rest, samples, "must SHRINK for namaste", true);
        Add("wrist height (m)", "height", rest, samples, "chest-level, not hips", false);
        Add("palm normal dot", "dot", rest, samples, "palms face each other (~ -1)", false);
        Add("wrist rotation (deg)", "wristrot", rest, samples, "wrist must turn", false);

        // NOTE on "palm normal dot": whether a correct namaste makes the two
        // normals parallel (+1) or anti-parallel (-1) depends on which
        // anatomical face the cross product picks per hand, so it is reported
        // (see values) but deliberately NOT asserted. The sign-independent
        // checks are the separation shrink and the wrist-rotation floor.

        _report.AppendLine();
        _report.AppendLine("=== FINDINGS ===");
        if (_findings.Count == 0) _report.AppendLine("none");
        else foreach (var f in _findings) _report.AppendLine(f);

        Write(_report.ToString());
        Debug.Log("[SignPlaybackProbe]\n" + _report);

        _mapper.CaptureDiagnostics = false;
        _animator.enabled = animWas;

        Assert.IsEmpty(_findings, "Sign playback check failed (see report).");
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

    private Dictionary<string, float> Measure()
    {
        var d = new Dictionary<string, float>();
        Transform lw = Bone(HumanBodyBones.LeftHand), rw = Bone(HumanBodyBones.RightHand);
        if (lw == null || rw == null) return d;

        d["sep"] = Vector3.Distance(lw.position, rw.position);
        d["height"] = 0.5f * (lw.position.y + rw.position.y);
        d["dot"] = Vector3.Dot(PalmNormal(lw), PalmNormal(rw));

        // Real wrist rotation vs the captured rest rotations.
        float lwr = Quaternion.Angle(_restLocalL, lw.localRotation);
        float rwr = Quaternion.Angle(_restLocalR, rw.localRotation);
        d["wristrot"] = Mathf.Max(lwr, rwr);

        float rot = 0f;
        var diag = _mapper.LastDiagnostics;
        if (diag != null)
            foreach (var kv in diag)
                if (kv.Value.appliedDeg > rot) rot = kv.Value.appliedDeg;
        d["rot"] = rot;

        d["lx"] = lw.position.x; d["rx"] = rw.position.x;
        d["ly"] = lw.position.y; d["ry"] = rw.position.y;
        d["lz"] = lw.position.z; d["rz"] = rw.position.z;
        return d;
    }

    private Transform Bone(HumanBodyBones b)
    {
        try { return _animator.GetBoneTransform(b); } catch { return null; }
    }

    private Vector3 PalmNormal(Transform hand)
    {
        bool right = hand.name.ToLowerInvariant().Contains("right");
        Transform idx = Bone(right ? HumanBodyBones.RightIndexProximal : HumanBodyBones.LeftIndexProximal);
        Transform mid = Bone(right ? HumanBodyBones.RightMiddleProximal : HumanBodyBones.LeftMiddleProximal);
        if (idx == null || mid == null) return Vector3.forward;
        Vector3 n = Vector3.Cross(idx.position - hand.position, mid.position - hand.position);
        return n.sqrMagnitude < 1e-10f ? Vector3.forward : n.normalized;
    }

    private void Add(string label, string key, Dictionary<string, float> rest,
                     List<Dictionary<string, float>> samples, string expect, bool shrink)
    {
        float min = float.MaxValue, max = float.MinValue;
        foreach (var s in samples)
        {
            float v = s.ContainsKey(key) ? s[key] : 0f;
            min = Mathf.Min(min, v); max = Mathf.Max(max, v);
        }
        float rv = rest.ContainsKey(key) ? rest[key] : 0f;

        string verdict;
        if (samples.Count == 0) verdict = "no samples";
        else if (shrink) verdict = min < rv - 0.02f ? "ok (hands come together)" : "FAIL (hands do not come together)";
        else verdict = "see values";

        if (shrink && samples.Count > 0 && !(min < rv - 0.02f))
            _findings.Add("HANDS-APART " + label + ": rest " + rv.ToString("0.00")
                + " clip " + min.ToString("0.00") + ".." + max.ToString("0.00") + " -- " + expect);

        _report.AppendLine(string.Format("{0,-24} {1,-8} {2,-21} {3}",
            label, rv.ToString("0.00"),
            samples.Count == 0 ? "-" : min.ToString("0.00") + ".." + max.ToString("0.00"),
            verdict));
    }

    private static void Write(string text)
    {
        string[] tries = {
            Path.Combine(Application.dataPath, "..", "sign_playback_report.txt"),
            Path.Combine(Application.persistentDataPath, "sign_playback_report.txt"),
        };
        foreach (var p in tries)
        {
            try { File.WriteAllText(p, text); Debug.Log("[SignPlaybackProbe] report -> " + p); return; }
            catch { }
        }
    }
}