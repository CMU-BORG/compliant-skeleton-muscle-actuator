import argparse
from tracker import VidTracker

def main():
    parser = argparse.ArgumentParser(description="Process video")
    parser.add_argument("name",  type=str)

    args = parser.parse_args()
    name = vars(args)["name"]
    
    tracker = VidTracker(name)
    tracker.load()
    tracker.track()

if(__name__ == "__main__"):
    main()