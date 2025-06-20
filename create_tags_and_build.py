import argparse
import os
import subprocess
import sys


# 1.2.3, 1 is major, 2 is minor, 3 is hotfix
RELEASE_TYPE_HOTFIX = "hotfix"
RELEASE_TYPE_MINOR = "minor"
RELEASE_TYPE_MAJOR = "major"
# order of list should match order of numbers in version
SUPPORTED_RELEASE_TYPES = [RELEASE_TYPE_MAJOR, RELEASE_TYPE_MINOR, RELEASE_TYPE_HOTFIX]


def get_git_branch():
    result = subprocess.run(['git', 'branch', '--show-current'], stdout=subprocess.PIPE, check=True)
    branch_name = result.stdout.decode().strip()
    return branch_name

def get_latest_git_tag():
    result = subprocess.run(['git', 'tag', '--sort=-creatordate'], stdout=subprocess.PIPE, check=True)
    tags = result.stdout.decode().split("\n")
    tag_name = (tags[0].strip())[1:]
    return tag_name

def create_tag_with_annotation(tag_name: str, annotation: str):
    result = subprocess.run(['git', 'tag', tag_name, '-m', annotation], stdout=subprocess.PIPE, check=True)
    result = subprocess.run(['git', 'push', 'origin', tag_name], stdout=subprocess.PIPE, check=True)


def build_wheel():
    result = subprocess.run(['hatch', 'build'], stdout=subprocess.PIPE, check=True)
    out = result.stdout.decode()
    print(out.split("\n")[-2:])



def main(release_type: str, tag_and_push: bool = False):

    if release_type not in SUPPORTED_RELEASE_TYPES:
        raise ValueError(f"release_type not supported, see list of supported types: {SUPPORTED_RELEASE_TYPES}")

    # first we check that we are on main or master, and warn users if not
    branch_name = get_git_branch()
    if tag_and_push and branch_name not in ["main", "master"]:
        raise ValueError("Must have master or main checked out to tag and build. Please verify you are on the right branch.")
    # # then we get the latest tag and increment it
    latest_tag = get_latest_git_tag()
    new_tag = None
    for i in range(len(SUPPORTED_RELEASE_TYPES)):
        if SUPPORTED_RELEASE_TYPES[i] != release_type:
            continue
        nums = latest_tag.split(".")
        nums = [int(n) for n in nums]
        nums[i] = nums[i] + 1
        str_nums = [str(n) for n in nums]
        new_tag = "v" + ".".join(str_nums)
    if tag_and_push:
        print("please enter an annotation for the new tag--a sentence or two about the new features or bugfixes since the last version. Press [enter] when done.")
        annotation = input()
        print(f"prior tag is {latest_tag}, new tag to create will be {new_tag}, annotation will be '{annotation}'. Is this correct (y/n)? Please verify before we push.")
        correct_tag = input()
        if correct_tag != "y":
            raise RuntimeError
        create_tag_with_annotation(new_tag, annotation)
    else:
        print(f"the tag we would have created would have been: {new_tag}, prior tag {latest_tag}")

    # now that we've tagged, we can build the wheel file
    print("building wheel...")
    build_wheel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release_type", choices=SUPPORTED_RELEASE_TYPES,
                        help=f"which number in the version to increment, specified by choosing from {SUPPORTED_RELEASE_TYPES}")
    parser.add_argument("-t", "--tag_and_push", help="This is for when we are ready for a release. If this flag is included a new git tag will be created incrementing the last tag and it will be pushed to the remote repository before building the wheel.",
                        action="store_true")
    args = parser.parse_args()
    main(args.release_type, args.tag_and_push)