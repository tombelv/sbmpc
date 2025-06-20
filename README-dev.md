# developper guide

## Building the Package
We synchronize our package version with git tags. We have a convenience script to create and push the git tags to the repo, and to then build the python package with the same tag version number. To use this, go to the root of the repo, and then run the below. The single argument is what number to increment, from the prior version. For example, if the prior version is 1.0.3, hotfix will go to 1.0.4, minor will go to 1.1.3, major will go to 2.0.3. Default to hotfix, when making breaking changes go bigger. This will not tag locally or push a tag, it's simply to test the build: `python create_tags_and_build.py hotfix`.

To actually push the new tag to git (local and remove), for cases when we are ready for a release, use the script with the below flag: `python create_tags_and_build.py hotfix --tag_and_push`.

If you are iteratively testing new builds with the same version number before a release, you will want to force install the wheel otherwise pip will ignore the same version number. When doing this take caution that it doesn't re-install jax and potentially mess up your setup: `pip install ./dist/sbmpc-0.0.0-py3-none-any.whl --force-reinstall --no-deps`.