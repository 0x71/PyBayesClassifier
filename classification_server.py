# Copyright (C) 2016-2017  Nils Rogmann.
# This file is part of PyBayesClassifier.
# See the file 'docs/LICENSE' for copying permission.

import os
import sys
import logging
import signal
import socket
import argparse
import ssl
import json
import math
import ConfigParser
from SimpleXMLRPCServer import SimpleXMLRPCServer

# Specify the number of meta rows that will be removed by bayes classifier
meta_rows = 4

BIND_IP = "127.0.0.1"
BIND_PORT = 8000

try:
    from lib.common.constants import APP_ROOT, TRAIN_RATIO, TEST_RATIO, CLASS_THRESHOLD_WEIGHT
    from lib.core.startup import init_logging, interrupt_handler
    from bayes_classify_one_json import BayesOneClass
    from lib.core.database import Database
except ImportError as e:
    sys.exit("ERROR: Missing library: {0}".format(e))

log = logging.getLogger()
signal.signal(signal.SIGINT, interrupt_handler)

class ClassificationServer(object):
    def __init__(self):
        log.info("Listener started.")
        #self.bayes_classifier = None

        # Init bayes
        #self._init_classifier()

        self.database = Database()

        self.sessions = {}

    def init_session(self, timestamp, tag):
        session_id = self.database.createSession(timestamp, tag)
        log.debug("New Session: %s " % session_id)
        self.sessions[session_id] = {"bayes": self._init_classifier()}

        return session_id

    def stop_session(self, timestamp, session_id):
        self.database.stopSession(timestamp, session_id)
        self.sessions.pop(session_id)
        log.debug("Session %s stopped." % session_id)

    def _init_classifier(self):
        return BayesOneClass()
    
    def set_baseline(self, baseline_data, session_id):
        log.info("New baseline data received.")
        #log.info(baseline_data)
        self.sessions[session_id]["bl_data"] = json.loads(baseline_data)

        bayes_classifier = self.sessions[session_id]["bayes"]
        log.info("Calculating baseline summary.")
        # Split into training and testing datasets according to split ratio
        train, test, validate = bayes_classifier.split_dataset(self.sessions[session_id]["bl_data"], TRAIN_RATIO, TEST_RATIO)
        log.info("Split %d rows into train with %s, test with %s and validate with %s." % (len(self.sessions[session_id]["bl_data"]), len(train), len(test), len(validate)))

        ### Train
        summary, counted = bayes_classifier.summarize_by_class(train)
        self.sessions[session_id]["bl_summary"] = summary
        log.info('Summary by class value: %s' % self.sessions[session_id]["bl_summary"])

        ### Test
        '''
        stream_probabilities = bayes_classifier.get_stream_probabilities(self.sessions[session_id]["bl_summary"], test) 

        ## Protocol probabilities
        mean_probabilities = bayes_classifier.get_mean_probabilities(stream_probabilities, counted)
        log.info("Mean probabilities: %s" % mean_probabilities)
        self.sessions[session_id]["bl_probs"] = mean_probabilities

        # Calculate thresholds based on mean probabilities for each stream
        for dkey, dval in mean_probabilities.iteritems():
            bayes_classifier.thresholds[dkey] = dval ** CLASS_THRESHOLD_WEIGHT
        print ""
        log.info("New thresholds: %s" % bayes_classifier.thresholds)
        '''

        ## Calculate thresholds based on feature probabilities
        # Feature probabilities first
        feature_probabilities = bayes_classifier.get_feature_probabilities(self.sessions[session_id]["bl_summary"], test)

        # Calculate feature thresholds, protocol thresholds, and protocol probabilities
        self.sessions[session_id]["bl_fthreshold"], bayes_classifier.thresholds, self.sessions[session_id]["bl_probs"] = bayes_classifier.get_thresholds(feature_probabilities)

        # Split threshold exponent and mantissa
        self.sessions[session_id]["bl_fthreshold_exp"], bayes_classifier.exp_thresholds = bayes_classifier.split_thresholds(self.sessions[session_id]["bl_fthreshold"])

        #log.info("PROTOCOL EXP: %s" % protocol_exp) 

        #log.info("Feature thresholds: %s" % self.sessions[session_id]["bl_fthreshold"])
        #log.info("Protocol probabilities: %s" % self.sessions[session_id]["bl_probs"])

        ### Validate
        predictions = bayes_classifier.get_predictions(self.sessions[session_id]["bl_summary"], validate)
        accuracy = bayes_classifier.get_accuracy(validate, predictions)
        self.sessions[session_id]["bl_accuracy"] = accuracy

        # Calculate threshold based on mean of predicted test values
        mean_predictions = bayes_classifier.get_mean_predictions(predictions, counted)

        log.info('Mean Predictions: %s' % mean_predictions)
        log.info('Accuracy: %s' % accuracy)

        self.database.addBaseline(session_id,
                                  self.sessions[session_id]["bl_summary"], # Baseline
                                  self.sessions[session_id]["bl_accuracy"], # Baseline self check
                                  bayes_classifier.thresholds, # Calculated protocol thresholds
                                  self.sessions[session_id]["bl_fthreshold"], # Calculated feature thresholds
                                  self.sessions[session_id]["bl_fthreshold_exp"], # Extracted feature threshold exponents
                                  bayes_classifier.exp_thresholds) # Extracted protocol threshold exponents

        print "Feature Thresholds"
        print self.sessions[session_id]["bl_fthreshold_exp"]

        #print self.sessions
        #print self.sessions[-1].keys()

        return len(self.sessions[session_id]["bl_data"])

    def classify_streams(self, stream_data, session_id):
        log.info("New stream data received.")

        stream_data = json.loads(stream_data)

        bayes_classifier = self.sessions[session_id]["bayes"]

        log.info("EXP THRESHOLDS")
        log.info(bayes_classifier.exp_thresholds)

        classifications = bayes_classifier.get_predictions(self.sessions[session_id]["bl_summary"], stream_data)

        ## Database update
        hosts = []
        for stream in stream_data.values():

            # Filter streams with very low probability
            try:
                protocol = stream["meta"]["type"]
                small_threshold = self.sessions[session_id]["bl_probs"][protocol] ** (CLASS_THRESHOLD_WEIGHT[protocol]/2.0)
                if True:
                #if stream["classify"]["prob"] < small_threshold:
                    # Hosts
                    if stream["meta"]["h_ip"] not in hosts:
                        host_id = self.database.addHost(stream["meta"]["h_ip"], stream["meta"]["h_id"], session_id)
                        hosts.append(stream["meta"]["h_ip"])
                    else:
                        host_id = self.database.getHost(stream["meta"]["h_ip"])

                    # Streams
                    stream_id = self.database.addStream(stream["meta"]["type"])
                    self.database.addStreamToHost(session_id, stream_id, host_id, stream)
            except KeyError:
                log.warning("No threshold specified for %s protocol." % type)
        return len(stream_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--debug", help="Display debug messages", action="store_true", required=False)
    args = parser.parse_args()

    # Start console and file logging
    init_logging()
    
    cfg = ConfigParser.ConfigParser()

    try:
        enable_ssl = False
        cfg.read(os.path.join(APP_ROOT, "conf", "server.conf"))
        ssl_support = cfg.get("security","ssl")
        if ssl_support == "True":
            enable_ssl = True
        else:
            log.warning("SSL disabled.")

    except:
        sys.exit("ERROR: Reading 'server.conf'")

    if args.debug:
        log.setLevel(logging.DEBUG)

    try:
        log.info("Starting Listener on %s:%s ..." % (BIND_IP, BIND_PORT))

        # Disable DNS lookup, by Scott D.
        def FakeGetFQDN(name=""):
            return name

        socket.getfqdn = FakeGetFQDN
        
        server = SimpleXMLRPCServer((BIND_IP,BIND_PORT), logRequests=False, allow_none=True)
        if enable_ssl == True:
            server.socket = ssl.wrap_socket(server.socket, keyfile='key.pem', certfile='cert.pem', server_side=True, cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_SSLv23)
        server.register_instance(ClassificationServer())
        server.serve_forever()

    except KeyboardInterrupt:
        server.shutdown()        