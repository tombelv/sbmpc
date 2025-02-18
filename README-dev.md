# developper guide

## Building the Package
We synchronize our package version with git tags. We have a convenience script to create and push the git tags to the repo, and to then build the python package with the same tag version number. To use this, go to the root of the repo, and then run the below. The single argument is what number to increment, from the prior version. For example, if the prior version is 1.0.3, hotfix will go to 1.0.4, minor will go to 1.1.3, major will go to 2.0.3. Default to hotfix, when making breaking changes go bigger.

```
python create_tags_and_build.py hotfix
```