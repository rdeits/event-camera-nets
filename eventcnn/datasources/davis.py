import os
import zipfile
import requests
import numpy as np
import pandas as pd
from .eventblock import EventBlock
from collections import namedtuple, OrderedDict


DAVIS_URL = "http://rpg.ifi.uzh.ch/datasets/davis/"

Format = namedtuple("Format",
                    ["filename",
                     "columns"])

event_format = Format("events.txt",
                      OrderedDict([("time", np.float64),
                                   ("x", np.uint8),
                                   ("y", np.uint8),
                                   ("polarity", np.bool)]))

groundtruth_format = Format("groundtruth.txt",
                            OrderedDict([("time", np.float64),
                                         ("px", np.float64),
                                         ("py", np.float64),
                                         ("pz", np.float64),
                                         ("qx", np.float64),
                                         ("qy", np.float64),
                                         ("qz", np.float64),
                                         ("qw", np.float64)]))


def download_file(url, path):
    response = requests.get(url, stream=True)
    with open(path, "wb") as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)


class DavisDataset:
    def __init__(self, store_path):
        self.store = pd.HDFStore(store_path)

    @staticmethod
    def read_csv(input_folder_path, store_path):
        print("Loading events")
        events = pd.read_csv(
                     os.path.join(input_folder_path,
                                  event_format.filename),
                     sep=" ",
                     header=None,
                     names=event_format.columns.keys(),
                     dtype=event_format.columns)
        print("Loading groundtruth")
        groundtruth = pd.read_csv(
                          os.path.join(input_folder_path,
                                       groundtruth_format.filename),
                          sep=" ",
                          header=None,
                          names=groundtruth_format.columns.keys(),
                          dtype=groundtruth_format.columns)

        with pd.HDFStore(store_path) as store:
            print("Storing groundtruth")
            store.append("groundtruth",
                         groundtruth,
                         format="table",
                         data_columns=True)
            print("Storing events")
            store.append("events",
                         events,
                         format="table",
                         data_columns=True)
        print("done storing")
        return DavisDataset(store_path)

    @staticmethod
    def named_dataset(name, data_folder="./data/davis"):
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        h5_path = os.path.join(data_folder, name + ".h5")
        if os.path.exists(h5_path):
            return DavisDataset(h5_path)
        else:
            zip_path = os.path.join(data_folder, name + ".zip")
            if not os.path.exists(zip_path):
                url = DAVIS_URL + "/" + name + ".zip"
                print("Downloading zipped dataset from: {}".format(url))
                download_file(url, zip_path)
            zf = zipfile.ZipFile(zip_path)
            print("Extracting zipped data")
            zf.extractall(os.path.join(data_folder, name))
            return DavisDataset.read_csv(os.path.join(data_folder,
                                                      name),
                                         os.path.join(data_folder,
                                                      name + ".h5"))

    @property
    def num_events(self):
        return self.store.root.events.table.nrows

    @property
    def num_groundtruth(self):
        return self.store.root.groundtruth.table.nrows

    def select_events(self, start_index, stop_index):
        assert start_index >= 0
        assert stop_index >= start_index
        assert stop_index < self.num_events
        return self.store.select("events",
                                 start=start_index,
                                 stop=stop_index)

    def event_block(self, start_index, stop_index):
        events = self.select_events(start_index, stop_index)
        start_time = events.iloc[0].time
        end_time = events.iloc[-1].time
        groundtruth_before = self.store.select(
            "groundtruth",
            "time <= start_time").iloc[-1]
        groundtruth_after = self.store.select(
            "groundtruth",
            "time >= end_time").iloc[0]
        delta_position = np.asarray(
            groundtruth_after[["px", "py", "pz"]] -
            groundtruth_before[["px", "py", "pz"]])
        delta_time = (groundtruth_after.time -
                      groundtruth_before.time)
        event_block_delta_position = (
            delta_position * (end_time - start_time) / delta_time)
        return EventBlock(events, event_block_delta_position)



