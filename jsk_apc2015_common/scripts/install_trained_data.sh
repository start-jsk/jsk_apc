#!/bin/sh

jsk_data get 2015-10-18-13-57-11_berkeley_dataset_bof.pkl.gz
mv 2015-10-18-13-57-11_berkeley_dataset_bof.pkl.gz `rospack find jsk_apc2015_common`/trained_data/berkeley_dataset_bof.pkl.gz

jsk_data get 2015-10-18-13-57-27_berkeley_dataset_lgr.pkl.gz
mv 2015-10-18-13-57-27_berkeley_dataset_lgr.pkl.gz `rospack find jsk_apc2015_common`/trained_data/berkeley_dataset_lgr.pkl.gz
