# Copyright (C) 2016-2017  Nils Rogmann.
# This file is part of PyBayesClassifier.
# See the file 'docs/LICENSE' for copying permission.

import os
import sys
import logging
import signal
import csv
import random
import math
import argparse
import json
from setuptools.dist import Feature

# Specify the number of meta rows that will be removed by bayes classifier
meta_rows = 4

try:
    from lib.common.constants import TRAIN_RATIO, TEST_RATIO, THRESHOLD, CLASS_THRESHOLD_WEIGHT, CLASS_THRESHOLD_WEIGHT_OLD
    from lib.core.startup import init_logging, interrupt_handler
except ImportError as e:
    sys.exit("ERROR: Missing library: {0}".format(e))
    
log = logging.getLogger()
signal.signal(signal.SIGINT, interrupt_handler)

class BayesOneClass():
    def __init__(self):
        log.info("BayesOneClass classifier started.")
        self.thresholds = {} # protocol thresholds
        self.exp_thresholds = {} # exponent separated protocol thresholds

    def is_float(self, value):
        try:
            num = float(value)
        except ValueError, Type:
            return False
        return True
    
    def is_int(self ,value):
        try:
            num = int(value)
        except ValueError:
            return False
        return True
      
    def try_numeric(self, line):
        ''' Automatically convert values in list to float, int or string '''
        try:
            tmp = []
            for val in line:
                if self.is_int(val):
                    tmp.append(int(val))
                elif self.is_float(val):
                    tmp.append(float(val))
                else:
                    tmp.append(str(val))
            return tmp
        except ValueError:
            pass
      
    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))
    
    def variance(self,numbers):
        try:
            avg = self.mean(numbers)
            return sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        except ZeroDivisionError:
            return THRESHOLD
    
    def stdev(self,numbers):
        return math.sqrt(self.variance(numbers))

    def frexp10(self, decimal):
        try:
            if decimal > 0.0:
                # Source: http://www.gossamer-threads.com/lists/python/python/867117
                val = math.log10(decimal)
                exp = math.ceil(val) if val > 0 else math.floor(val)
                return decimal / 10 ** exp, exp
            else:
                return 1.0, -310
        except Exception as e:
            log.error("MATH ERROR: %s" % e)
            log.error("Decimal: %s" % decimal)
            log.error("Val: %s" % val)
            log.error("Exp: %s" % exp)
            log.error("Val: %s" % val)
            return 1.0, -310
            

    def load_csv(self,file_name):
        lines = []
        with open(file_name, 'rb') as dataset:
            lines = list(csv.reader(dataset, quoting=csv.QUOTE_NONE))
            for i in range(0,len(lines)):
                # Try to guess value types line by line and value by value
                lines[i] = self.try_numeric(lines[i])
    
        return lines
    
    def load_json(self,file_name):
        data = {}
        with open(file_name, 'rb') as dataset:
            data = json.load(dataset)
        return data
    
    def filter_columns(self, dataset, columns):
        ''' Filter columns (features) according to input args. '''
        for line in dataset:
            for column in reversed(columns):
                try:
                    line.pop((meta_rows-1)+int(column))
                except IndexError as e:
                    log.error("Index Error: %s" % e)
                    sys.exit(1)
    
        return dataset
    
    def split_dataset(self, dataset, train_ratio, test_ratio):
        ''' Split dataset in train, test and validation data. '''
        # calculate training dataset size
        train_size = int(len(dataset) * train_ratio)
        train_set = {}
        #log.info("Training dataset size: %d", train_size)

        validation_set = dict(dataset)
        #log.info(testset)
        while len(train_set) < train_size:
            index = random.choice(validation_set.keys())
            train_set[index] = validation_set.pop(index)

        test_size = int(len(dataset) * test_ratio)
        test_set = {}
        #log.info("Test dataset size: %d", test_size)

        while len(test_set) < test_size:
            index = random.choice(validation_set.keys())
            test_set[index] = validation_set.pop(index)

        return [train_set, test_set, validation_set]
    
    def separate_by_class(self, dataset):
        ''' Separate by class based on protocol type. '''
        #print dataset
        separated = {}
        counted = {}
        counted['sum'] = len(dataset)
        for dkey, dval in dataset.iteritems():
            type = dval["meta"]["type"]
            if type not in separated:
                log.info("Creating new Class: %s" % type)
                separated[type] = []
                counted[type] = 0
            
            separated[type].append(dataset[dkey])
            counted[type] = counted[type] + 1

        log.info(counted)
        return separated, counted
    
    def summarize(self, dataset):
        ''' Calculate mean and stdev for each row (needed for gauss distribution). '''
        extracted = {}
        for val in dataset:
            for dkey, dval in val["data"].iteritems():
                if dkey not in extracted:
                    extracted[dkey] = []

                extracted[dkey].append(dval)
        
        #log.info("Extracted values: %s" % extracted)

        #summary = {dkey: [self.mean(dval), self.stdev(dval)] for dkey, dval in extracted.iteritems()}

        summary = {}
        for dkey, dval in extracted.iteritems():
            if self.is_float(dval[0]) or self.is_int(dval[0]):
                summary[dkey] = {"value": [self.mean(dval), self.stdev(dval)], "type":"num"}
            else:
                occurences = {}
                total = 0
                for val in dval:
                    if val in occurences.keys():
                        occurences[val] +=1
                    else:
                        occurences[val] = 1
                    total += 1
                        
                for key, val in occurences.iteritems():
                    occurences[key] = val/float(total)
                        
                summary[dkey] = {"value": occurences, "type":"cat"}
        return summary
    
    def summarize_by_class(self, dataset):
        separated, counted = self.separate_by_class(dataset)
        summaries = {}
        for class_value, instances in separated.iteritems():
            # print instances
            summaries[class_value] = self.summarize(instances)

        return summaries, counted
    
    def gauss_distribution(self, val, mean, stdev):
        try:
            exponent = math.exp(-(math.pow(val-mean, 2)) / (2 * math.pow(stdev,2)))
            return (1 / (math.sqrt(2*math.pi*math.pow(stdev,2))) * exponent)
        except ZeroDivisionError:
            #log.info("Division by zero (probably stdev is zero). Using Threshold %s" % THRESHOLD)
            return THRESHOLD

    def calculate_class_probabilities(self, summaries, input_vector, details=False):
        ''' Calculate class probability for according to prototcol type '''
        detail_dict = {}
        exp_probability_dict = {}

        # Get protocol type
        stream_class = input_vector["meta"]["type"]
        if details:
            if stream_class not in detail_dict:
                detail_dict[stream_class] = {}
                exp_probability_dict[stream_class] = {}
        probability = 1
        exp_probability = 0
        mantissa = 1
        
        try:
            # Take all streams of the same protocol
            class_summary = summaries[stream_class]
            #print class_summary
        except KeyError:
            log.warning("Key %s not in training set." % stream_class)
            return 0

        # Iterate over protocol class summary
        for dkey, dval in class_summary.iteritems():
            
            # numerical feature
            if dval["type"] == "num":
                # Extract mean and stdev for each element in class summary
                mean, stdev = dval["value"]
            
                classify_me = input_vector["data"][dkey]
            
                gauss_val = self.gauss_distribution(classify_me, mean, stdev)
                probability = probability * gauss_val
                exp_val = self.frexp10(gauss_val)[1] # 'true_val' if 'condition' else 'false_value'
                mantissa = mantissa * self.frexp10(gauss_val)[0]
                exp_probability = exp_probability + exp_val
                if details:
                    detail_dict[stream_class][dkey] = gauss_val
                    exp_probability_dict[stream_class][dkey] = self.frexp10(gauss_val)
                    log.debug("%s: %s, mean: %s, stdev: %s: %s (%s)" % (dkey, classify_me, mean, stdev, gauss_val, exp_val))
            
            if dval["type"] == "cat":
                occurences = dval["value"]
                
                classify_me = input_vector["data"][dkey]
                
                
                if classify_me in occurences.keys():
                    rel_probability = occurences[classify_me]
                    probability = probability * rel_probability
                else:
                    probability = probability * THRESHOLD
                    rel_probability = 0
                    
                exp_val = self.frexp10(rel_probability)[1]
                mantissa = mantissa * self.frexp10(rel_probability)[0]
                exp_probability = exp_probability + exp_val
                if details:
                    detail_dict[stream_class][dkey] = rel_probability
                    exp_probability_dict[stream_class][dkey] = self.frexp10(rel_probability)
                    log.debug("%s: %s: %s (%s)" % (dkey, classify_me, rel_probability, exp_val))
                
                
                

        log.debug("")

        exp_probability = exp_probability + self.frexp10(mantissa)[1]

        input_vector["classify"] = {}
        input_vector["classify"]["prob"] = probability
        input_vector["classify"]["prob_details"] = detail_dict

        input_vector["classify"]["prob_exp"] = {"mantissa": mantissa, "exp": exp_probability}
        input_vector["classify"]["prob_details_exp"] = exp_probability_dict

        if details:
            return [stream_class, probability, exp_probability], detail_dict
        else:
            return [stream_class, probability, exp_probability]
                
    def predict(self, summaries, input_vector):
        ''' Predict class affiliation based on one class classification. '''
        probability, feature_details = self.calculate_class_probabilities(summaries, input_vector, details=True)
        try:
            #print "Probability: %s." % probability[2]
            #print "Threshold: %s." % self.exp_thresholds[input_vector["meta"]["type"]][1]
            #if probability[1] > self.thresholds[input_vector["meta"]["type"]]:
            if probability[2] > (self.exp_thresholds[input_vector["meta"]["type"]][1]):
                #print "MATCH"
                input_vector["classify"]["in_class"] = 1
                #log.info("[Stream %s] Normal stream: %s (%s)" % (input_vector["meta"]["_id"], probability[1], probability[2]))
                return (input_vector["meta"]["_id"], probability[0], 1, probability[1], probability[2]) # in class
            else:
                log.warning("[Stream %s] Data exfiltration detected: %s (%s)" % (input_vector["meta"]["_id"], probability[1], probability[2]))
                input_vector["classify"]["in_class"] = 0
                return (input_vector["meta"]["_id"], probability[0], 0, probability[1], probability[2]) # not in class
        except KeyError:
            log.warning("No threshold specified for %s." % probability[0])
            return (input_vector["meta"]["_id"], probability[0], -1, probability[1],probability[2]) # not in class

    def get_thresholds(self, feature_probabilities):
        mean_protocol_probability = {}
        feature_thresholds = {}
        protocol_thresholds = {}

        #log.info("FEATURES: %s" % feature_probabilities)
        for protocol_key, feature_list in feature_probabilities.iteritems():
            log.info("Feature list: %s" % feature_list)
            total = 1 # Total probability for each feature
            feature_thresholds[protocol_key] = {}
            protocol_thresholds[protocol_key] = {}
            for feature, value in feature_list.iteritems():
                if value < 1:
                    feature_thresholds[protocol_key][feature] = value ** CLASS_THRESHOLD_WEIGHT[protocol_key]
                else:
                    feature_thresholds[protocol_key][feature] = value ** -CLASS_THRESHOLD_WEIGHT[protocol_key]
                total = total * value
            mean_protocol_probability[protocol_key] = total
            protocol_thresholds[protocol_key] = total ** CLASS_THRESHOLD_WEIGHT[protocol_key]

        print ""
        log.info("Feature thresholds: %s" % feature_thresholds)
        log.info("New protocol thresholds: %s" % protocol_thresholds)

        return feature_thresholds, protocol_thresholds, mean_protocol_probability
    
    def split_thresholds(self, feature_thresholds):
        print ""
        #print feature_thresholds
        splitted_feature_thresholds = {}
        splitted_protocol_thresholds = {}
        for protocol_key, feature_list in feature_thresholds.iteritems():
            total_exp = 0
            total_mantissa = 1
            splitted_feature_thresholds[protocol_key] = {}
            splitted_protocol_thresholds[protocol_key] = {}
            print ""
            for feature, threshold in feature_list.iteritems():
                splitted_feature_thresholds[protocol_key][feature] = self.frexp10(threshold)

                total_mantissa = total_mantissa * splitted_feature_thresholds[protocol_key][feature][0]
                total_exp = total_exp + splitted_feature_thresholds[protocol_key][feature][1]

                #print protocol_key, feature, threshold, splitted_feature_thresholds[protocol_key][feature]

            # Improve exponent value: Add missing exponent value created during mantissa multiplication
            total_exp = total_exp+self.frexp10(total_mantissa)[1]

            # Normalize mantissa value
            total_mantissa = self.frexp10(total_mantissa)[0]
            splitted_protocol_thresholds[protocol_key] = (total_mantissa, total_exp)
            #print total_mantissa, total_exp

        # return splitted_feature_thresholds, corrected mantissa, corrected exponent
        return splitted_feature_thresholds, splitted_protocol_thresholds

    def get_predictions(self, summaries, test_vector):
        predictions = []
        for dval in test_vector.values():
            result = self.predict(summaries, dval)
            if result:
                predictions.append(result)
        return predictions

    def get_stream_probabilities(self, summaries, test_vector):
        #print "TEST VECTOR:", test_vector
        probabilities = []
        for dval in test_vector.values():
            result = self.calculate_class_probabilities(summaries, dval, details=False)
            probabilities.append(result)

        return probabilities

    def get_feature_probabilities(self, summaries, test_vector):
        #print ""
        #print log.info("Test Vector: %s" % test_vector)
        feature_summary = {}
        for dval in test_vector.values():
            result, feature_details = self.calculate_class_probabilities(summaries, dval, details=True)
            
            #log.info("Feature details: %s" % feature_details)

            ### Extract feature details for mean calculation per feature
            # Get current protocol name
            protocol = feature_details.keys()[0] # There is only one protocol in each loop
            if protocol not in feature_summary: # Add new protocol to feature summary
                feature_summary[protocol] = feature_details[protocol]
                for feature, value in feature_details[protocol].iteritems():
                    feature_summary[protocol][feature] = [value]
            else: # Append feature probability to feature key for each protocol
                for feature_key in feature_summary[protocol].iterkeys():
                    feature_summary[protocol][feature_key].append(feature_details[protocol][feature_key])

        #print ""
        #log.info("Feature Summary: %s" % feature_summary)
        # Calculate mean probability for each feature
        for protocol_key, feature_list in feature_summary.iteritems():
            total = 1
            for feature, value in feature_list.iteritems():
                feature_summary[protocol_key][feature] = self.mean(value)
                
        #log.info("Feature Summary: %s" % feature_summary)

        return feature_summary

    def get_accuracy(self, test_vector, predictions):
        ''' Predict prediction accuracy for each protocol. '''
        stream_classes = {} # Calculate accuracy for each protocol
        count_classes = {}
        correct = 0
        for x in range(0,len(test_vector)):
            if predictions[x][2] == 1: # in class
                try:
                    stream_classes[predictions[x][1]] += 1 # e.g. stream_classes['dns'][0] +=1
                except KeyError:
                    # Key does not exist yet
                    stream_classes[predictions[x][1]] = 1
                correct += 1
                    
            try:
                count_classes[predictions[x][1]] += 1
            except KeyError:
                count_classes[predictions[x][1]] = 1

        log.info("True positves: %s/%s." % (correct, len(test_vector)))
        for stream_class, stream_value in stream_classes.iteritems():
            stream_classes[stream_class] = stream_value/float(count_classes[stream_class]) * 100.0

        return {'classes': stream_classes, 'correct': correct, 'total': len(test_vector)}

    def get_mean_predictions(self, predictions, counted):
        mean_predictions = {}
        #print predictions
        for sid, type, value, prediction, prediction_exp in predictions:
            #print predictions
            try:
                mean_predictions[type] += prediction
            except KeyError:
                mean_predictions[type] = prediction

        for key in mean_predictions.iterkeys():
            mean_predictions[key] = mean_predictions[key]/counted[key]

        return mean_predictions

    def get_mean_probabilities(self, probabilities, counted):
        mean_probabilities = {}
        #print probabilities
        for key, probability in probabilities:
            try:
                mean_probabilities[key] += probability
            except KeyError:
                mean_probabilities[key] = probability

        for key in mean_probabilities.iterkeys():
            mean_probabilities[key] = mean_probabilities[key]/counted[key]

        return mean_probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--learnfile", help="Training and test file", type=str, required=True)
    parser.add_argument("-c","--classify", help="Classification file", type=str, required=False)
    parser.add_argument("--filter", help="Filter columns; format: c1,c2,c3", type=str, required=False)
    args = parser.parse_args()

    # Start console and file logging
    init_logging()
    log.setLevel(logging.DEBUG)
    
    bayes_classifier = BayesOneClass()
    
    # Read and parse the data file
    file_name = args.learnfile
    dataset = bayes_classifier.load_json(file_name)
    log.info('Loaded data file %s with %d streams.' % (file_name, len(dataset)))
    #print dataset
    
    # Filter columns
    #if args.filter:
    #    log.info("Filtering columns: %s" % args.filter)
    #    dataset = bayes_classifier.filter_columns(dataset, args.filter.split(","))

    # Split into training and testing datasets according to split ratio
    train, test, validate = bayes_classifier.split_dataset(dataset, TRAIN_RATIO, TEST_RATIO)
    log.info("Split %d rows into train with %s, test with %s and validate with %s." % (len(dataset), len(train), len(test), len(validate)))
    
    # Train
    summary, counted = bayes_classifier.summarize_by_class(train)
    log.info('Summary by class value: %s' % summary)
    
    # Calculate threshold
    probabilities = bayes_classifier.get_stream_probabilities(summary, test, details=False)
    mean_probabilities = bayes_classifier.get_mean_probabilities(probabilities, counted)
    log.info("Mean probabilities: %s" % mean_probabilities)

    for dkey, dval in mean_probabilities.iteritems():
        bayes_classifier.thresholds[dkey] = dval ** CLASS_THRESHOLD_WEIGHT_OLD
    log.info("New thresholds: %s" % bayes_classifier.thresholds)

    # Validate
    predictions = bayes_classifier.get_predictions(summary, test)
    accuracy = bayes_classifier.get_accuracy(test, predictions)

    # Calculate threshold based on mean of predicted test values
    mean_predictions = bayes_classifier.get_mean_predictions(predictions, counted)

    log.info('Mean Predictions: %s' % mean_predictions)
    log.info('Accuracy: %s' % accuracy)
    
    # Classify
    if args.classify:
        dataset2 = bayes_classifier.load_json(args.classify)
        # Filter columns
        #if args.filter:
        #    dataset2 = bayes_classifier.filter_columns(dataset2, args.filter.split(","))
        classifications = bayes_classifier.get_predictions(summary, dataset2)
        log.info('Classification: %s' % classifications)
