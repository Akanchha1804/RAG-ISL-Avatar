
using UnityEngine;

public class AvatarRigInspector : MonoBehaviour
{
    public Animator animator;

    void Start()
    {
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }

        if (animator == null)
        {
            Debug.LogError("Animator not found.");
            return;
        }

        if (!animator.isHuman)
        {
            Debug.LogError("Avatar is not configured as Humanoid.");
            return;
        }

        Debug.Log("Humanoid avatar detected.");

        CheckBone(HumanBodyBones.LeftUpperArm);
        CheckBone(HumanBodyBones.LeftLowerArm);
        CheckBone(HumanBodyBones.LeftHand);

        CheckBone(HumanBodyBones.RightUpperArm);
        CheckBone(HumanBodyBones.RightLowerArm);
        CheckBone(HumanBodyBones.RightHand);

        CheckBone(HumanBodyBones.LeftIndexProximal);
        CheckBone(HumanBodyBones.RightIndexProximal);
    }

    void CheckBone(HumanBodyBones bone)
    {
        Transform boneTransform = animator.GetBoneTransform(bone);

        if (boneTransform != null)
        {
            Debug.Log(
                bone + " -> " + boneTransform.name
            );
        }
        else
        {
            Debug.LogWarning(
                bone + " is not mapped."
            );
        }
    }
}
