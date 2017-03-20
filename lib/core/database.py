# Copyright (C) 2016-2017  Nils Rogmann.
# This file is part of PyBayesClassifier.
# See the file 'docs/LICENSE' for copying permission.

import sys
import logging
from time import time
import json
import ConfigParser
import os

try:
    from sqlalchemy import create_engine
    from sqlalchemy import ForeignKey
    from sqlalchemy.orm import sessionmaker, scoped_session
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import Column, Integer, Float, String, Boolean
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm.exc import NoResultFound
    from sqlalchemy.ext import mutable
    from sqlalchemy import TypeDecorator
    from lib.common.constants import APP_ROOT
    #from sqlalchemy.types import JSON
except ImportError as e:
    sys.exit("ERROR: Missing library: {0}".format(e))
    
log = logging.getLogger(__name__)

cfg = ConfigParser.ConfigParser()
try:
    cfg.read(os.path.join(APP_ROOT, "conf", "database.conf"))
    conf_engine = cfg.get("connection","engine")

except:
    sys.exit("ERROR: Reading 'sniffer.conf'")

engine = create_engine(conf_engine, echo=False)
#Session = sessionmaker(bind=engine)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
#session = Session()

Base = declarative_base()

class JsonEncodedDict(TypeDecorator):
  """Enables JSON storage by encoding and decoding on the fly."""
  impl = String

  def process_bind_param(self, value, dialect):
    return json.dumps(value)

  def process_result_value(self, value, dialect):
    return json.loads(value)

mutable.MutableDict.associate_with(JsonEncodedDict)


class Database():
    def __init__(self):
        log.debug("Database started.")
        self.mysession = None
        self.hosts = []
        self.sessionhashosts = []   

    def __del__(self):
        log.debug("Database stopped.")

    def createSession(self, start, tag):
        log.debug("Creating session.")
        session = Session()
        log.debug("Done.")
        self.mysession = SniffSession(start,0, tag)
        log.debug("Session: %s" % self.mysession)
        
        session.add(self.mysession)
        session.commit()
        session_id = self.mysession.SessionID
        session.close()

        return session_id

    def stopSession(self, stop, session_id):
        session = Session()
        session.query(SniffSession).filter_by(SessionID=session_id).update({"EndTime": stop})
        session.commit()
        session.close()

    def addHost(self, ipaddr, identifier, session_id):
        try:
            session = Session()
            log.debug("ADDING HOST")
            self.hosts.append(SniffHost(ipaddr))
            session.add(self.hosts[-1])
            session.commit()
            log.debug("Host %s added." % ipaddr)
        except IntegrityError:
            log.debug("HOST ALREADY EXISTS.")
            session.rollback()

        host_id = self.addHostToSession(ipaddr, identifier, session_id, session)
        session.close()

        return host_id

    def addHostToSession(self, ipaddr, identifier, session_id, session):
        records = session.query(SniffHost).filter_by(IPAddr=ipaddr)
        host_id = records.one().HostID

        records = session.query(SessionHasHost).filter_by(SessionID=session_id, HostID=host_id)
        if len(records.all()) == 0:
            self.sessionhashosts.append(SessionHasHost(session_id, host_id, identifier))
            session.add(self.sessionhashosts[-1])
            session.commit()

        return host_id

    def getHost(self, ipaddr):
        session = Session()
        records = session.query(SniffHost).filter_by(IPAddr=ipaddr)
        host_id = records.one().HostID
        session.close()

        return host_id

    def addStream(self, type):
        session = Session()
        records = session.query(SniffStream).filter_by(Type=type)
        try:
            stream_id = records.one().StreamID
            session.close()
            return stream_id
        except NoResultFound:
            mystream = SniffStream(type)
            session.add(mystream)
            session.commit()
            session.close()
            return mystream.StreamID

    def addStreamToHost(self, session_id, stream_id, host_id, stream_data):
        session = Session()
        # extract meta information
        meta = stream_data["meta"]

        first_timestamp = meta["start"]
        last_timestamp = meta["end"]
        identifier = meta["_id"]

        # print stream_data

        stream_host = HostHasStream(session_id,
                                    host_id,
                                    stream_id,
                                    identifier,
                                    first_timestamp,
                                    last_timestamp,
                                    stream_data["meta"],
                                    stream_data["data"],
                                    stream_data["classify"]["prob"],
                                    stream_data["classify"]["prob_details"],
                                    stream_data["classify"]["prob_exp"],
                                    stream_data["classify"]["prob_details_exp"],
                                    stream_data["classify"]["in_class"])

        #print stream_host
        session.add(stream_host)
        session.commit()
        session.close()

    def addBaseline(self, session_id, baseline_summary, baseline_accuracy, threshold_summary, threshold_details, threshold_feature_exponents, threshold_protocol_exponents):
        log.debug("Test")
        session = Session()
        log.debug("Adding baseline summary and thresholds to database.")
        baseline = SniffBaseline(baseline_summary, baseline_accuracy, threshold_summary, threshold_details, threshold_feature_exponents, threshold_protocol_exponents)
        session.add(baseline)
        session.commit()

        session_has_baseline = SessionHasBaseline(session_id, baseline.BaselineID)
        session.add(session_has_baseline)
        session.commit()
        session.close()

class SniffSession(Base):
    __tablename__ = "SniffSession"

    SessionID = Column(Integer, primary_key=True, autoincrement=True)
    Tag = Column(String)
    StartTime = Column(Float)
    EndTime = Column(Float)

    def __init__(self, start=0, end=0, tag=""):
        self.Tag = tag
        self.StartTime = start
        self.EndTime = end

    def __repr__(self):
        return "<SniffSession(%s, %s, %s, %s)>" % (self.SessionID, self.Tag, self.StartTime, self.EndTime)

class SniffHost(Base):
    __tablename__ = "SniffHost"

    HostID = Column(Integer, primary_key=True, autoincrement=True)
    IPAddr = Column(String(45))

    def __init__(self, ipaddr=0):
        self.IPAddr = ipaddr

    def __repr__(self):
        return "<SniffHost(%s, %s, %s)>" % (self.HostID, self.IPAddr)

class SessionHasHost(Base):
    __tablename__ = "SniffSession_has_SniffHost"

    SessionID = Column(Integer, ForeignKey('SniffSession.SessionID'), primary_key=True)
    HostID = Column(Integer, ForeignKey('SniffHost.HostID'), primary_key=True)
    DisplayHostID = Column(Integer, unique=True)

    def __init__(self, session_id, host_id, display_id):
        self.SessionID = session_id
        self.HostID = host_id
        self.DisplayHostID = display_id

    def __repr__(self):
        return "<SessionHasHost(%s, %s, %s)>" % (self.SessionID, self.HostID, self.DisplayHostID)

class SniffStream(Base):
    __tablename__ = "SniffStream"

    StreamID = Column(Integer, primary_key=True, autoincrement=True)
    Type = Column(String(10), unique=True)

    def __init__(self,  type):
        self.Type = type

    def __repr__(self):
        return "<SniffStream(%s, %s, %s)>" % (self.SessionID, self.Type)

class HostHasStream(Base):
    __tablename__ = "SniffSession_has_SniffHost_has_SniffStream"

    SessionID = Column(Integer, ForeignKey('SniffSession_has_SniffHost.SessionID'), primary_key=True)
    HostID = Column(Integer, ForeignKey('SniffSession_has_SniffHost.HostID'), primary_key=True)
    StreamID = Column(Integer, ForeignKey('SniffStream.StreamID'), primary_key=True)
    LocalStreamID = Column(Integer, primary_key=True)
    StartTime = Column(Float)
    EndTime = Column(Float)
    Meta = Column(JsonEncodedDict)
    Features = Column(JsonEncodedDict)
    Probability = Column(String)
    ProbabilityDetails = Column(JsonEncodedDict)
    ExpProbability = Column(JsonEncodedDict)
    ExpProbabilityDetails = Column(JsonEncodedDict)
    Detected = Column(Boolean)

    def __init__(self, session_id, host_id, stream_id, local_id, start, end, meta, features, probability, prob_details, exp_probability, exp_prob_details, detected):
        self.SessionID = session_id
        self.HostID = host_id
        self.StreamID = stream_id
        self.LocalStreamID = local_id
        self.StartTime = start
        self.EndTime = end
        self.Meta = meta
        self.Features = features
        self.Probability = probability
        self.ProbabilityDetails = prob_details
        self.ExpProbability = exp_probability
        self.ExpProbabilityDetails = exp_prob_details
        self.Detected = detected

    def __repr__(self):
        return "<HostHasStream(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)>" % (self.SessionID, self.HostID, self.StreamID, self.LocalStreamID, self.StartTime, self.EndTime,
                                                                                self.Meta, self.Features, str(self.Probability), self.ProbabilityDetails, self.Detected)

class SniffBaseline(Base):
    __tablename__ = "SniffBaseline"

    BaselineID = Column(Integer, primary_key=True, autoincrement=True)
    Summaries = Column(JsonEncodedDict)
    Accuracy = Column(JsonEncodedDict)
    ThresholdSummary = Column(JsonEncodedDict)
    ThresholdDetails = Column(JsonEncodedDict)
    ExpThresholdDetails = Column(JsonEncodedDict)
    ExpThresholdSummary = Column(JsonEncodedDict)

    def __init__(self, summaries, accuracy, threshold_summary, threshold_details, threshold_feature_exp, threshold_protocol_exp):
        self.Summaries = summaries
        self.Accuracy = accuracy
        self.ThresholdSummary = threshold_summary
        self.ThresholdDetails = threshold_details
        self.ExpThresholdDetails = threshold_feature_exp
        self.ExpThresholdSummary = threshold_protocol_exp

    def __repr__(self):
        return "<SniffBaseline(%s, %s, %s, %s, %s)>" % (self.SessionID, self.Summaries, self.Accuracy, self.ThresholdSummary, self.ThresholdDetails)

class SessionHasBaseline(Base):
    __tablename__ = "SniffSession_has_SniffBaseline"

    SessionID = Column(Integer, ForeignKey('SniffSession.SessionID'), primary_key=True)
    BaselineID = Column(Integer, ForeignKey('SniffBaseline.BaselineID'), primary_key=True)

    def __init__(self, session_id, baseline_id):
        self.SessionID = session_id
        self.BaselineID = baseline_id

    def __repr__(self):
        return "<SessionHasBaseline(%s, %s)>" % (self.SessionID, self.BaselineID)