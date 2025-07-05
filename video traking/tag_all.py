from tracker import VidTracker

if(__name__ == "__main__"):
    VidTracker.checkFolders()
    VidTracker.batchTag(skipExisting=True)