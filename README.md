# **We Are Number One.**

A team based problem solving group at Imperial College London. Important information below, please read.

## Getting Started

To initialise the repository on your computer, in a chosen folder, perform,
```cmd
git clone https://github.com/Elfikk/TBPSWeAreNumberOne.git
```

This will pull the files on the main branch onto your computer and initiate a local copy of the repository in the target folder. Note that git will clone into a new subfolder in the directory you are in.

<br>

## Data Files
We have been given a lot of files for this project (3.20GB). To reduce the time it takes for the initial pull, we would like everyone to individually store their .csv files in a folder called `Data` in the `TBPSWeAreNumberOne` Folder locally on your PC. You should have the `TBPSWeAreNumberOne` folder if you have cloned the repository onto your PC.

As long as you name the folder `Data`, git will automatically ignore that folder when you push to a branch providing the file, [.gitignore](.gitignore), is on your PC. If it is not on your PC, please perform:
```cmd
git pull origin main
```
For all scripts made, please use this folder to load in data - this will make it extremely easy for everyone to run scripts without changing where files are located on their PC.

<br>

## Branching Conventions

1. Each subteam should have a main branch which they share, with the branch name simply being the purpose of the subteam.
2. Each member of a subteam should work on their own branch, with their initials being the branch name.

<br>

## Documentation

Each function that you write should be documented.

In the Wikis section of the GitHub repo, create a page with the function name, its parameters and outputs as per numpy documentation ([example](https://numpy.org/doc/stable/reference/generated/numpy.array.html)).

<br>

## Git 101

To switch to a different branch, perform,
```cmd
git switch BRANCH_NAME
```

To create your own branch, simply write,
```cmd
git branch BRANCH_NAME
```
By convention, your personal branch's name should be your initials. Note that you will not be automatically be switched to this branch on creation, so you must switch onto it.

To get the latest version of code from a branch, which will overwrite your local copy, perform,
```cmd
git pull
```

Be careful when committing your own changes. The safest route when making a commit to your own branch is to add individual files to the commit, via,
```cmd
git add RELATIVE_PATH_TO_REPO/FILENAME.EXTENSION
```
This can be tedious if you are working on a lot of files - if files that should not be pushed to your branch are in the [.gitignore](.gitignore) file, you may instead use,
```cmd
git add -A
```
For quick guidance on using [.gitignore](.gitignore), go [here](.https://www.atlassian.com/git/tutorials/saving-changes/gitignore).
Having added all the files you want to commit, perform,
```cmd
git commit -m "COMMIT_TITLE"
```
Then, to push the changes to GitHub, perform,
```cmd
git push
```
Hooray, you should be able to see your changes on your GitHub branch! 

<br>

Once you have completed work that should be shared with the rest of your subteam, you can merge your changes' with your subteam's branch by creating a pull request. On your branch, click,

![PullRequest](readme/PullRequest.png)

In the pull request, make sure to select the correct branch to merge with,
in the base menu,

![BaseBranch](readme/PullBaseBranch.png)

The same process applies to merging fully completed tasks of a subteam with the 
rest of the group. A subteam member should create a pull request to main. 
However, pass the pull request to Jarek Ciba for approval - easiest to contact
by Teams/Whatsapp.