# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import json
import logging
import os
import sys
import time

from utils import system


# The idea of this bot is that different machines/computers can send their status updates into a telegram channel.
#
# For the telegram bot to work you need to create a ".machines.json" file in your home directory with the content:
# {
#     "chat_id": "put your CHAT_ID in here",
#     "machines": {
#         "hostname1": "token for hostname1",
#         "hostname2": "token for hostname2"
#     }
# }
#
# E.g. an example would be
# {
#     "chat_id": "-2323423453,",
#     "machines": {
#         "gandalf": "4536536:3495890457340957",
#         "frodo": "3904285:2395453344534542"
#     }
# }
#
# Find out about how to obtain the required chat_id and tokens at https://core.telegram.org/bots#6-botfather
#
# The chat_id will represent a channel you send your messages into. The host names correspond to machine names
# (before any occurences of domains). For instance, for a host peter.pan.my-university.com the used lookup name
# will be peter. The IDENTIFIERS correspond to specific bots (that you should add to your channel.


class Bot:
    def __init__(self, filename, flush_secs):
        self.chat_id = None
        self.token = None
        self.last_time = 0
        self.message = ""
        self.flush_secs = flush_secs
        # deactivate stupid error messages
        telegram_logger = logging.getLogger("telegram")
        telegram_logger.setLevel(logging.ERROR)
        telegram_logger.propagate = True
        # load tokens
        self.initialize(filename)

    def initialize(self, filename):
        if not os.path.isfile(filename):
            logging.info("Could not find {}".format(filename))
        else:
            logging.info("Loading Telegram tokens from {}".format(filename))
            bots = None
            with open(filename, "r") as f:
                lines = f.readlines()
                try:
                    bots = json.loads(''.join(lines), encoding='utf-8')
                except Exception:
                    raise ValueError('Could not read %s. %s' % (filename, sys.exc_info()[1]))

            if bots is not None:
                hostname = system.hostname()
                logging.value("Found Host: ", hostname)
                if "chat_id" in bots.keys():
                    self.chat_id = bots["chat_id"]
                if "machines" in bots.keys():
                    machines = bots["machines"]
                    if hostname in machines.keys():
                        self.token = machines[hostname]
                        if self.token is not None:
                            try:  # try out once
                                from telegrambotapiwrapper import Api
                                Api(token=self.token)
                            except Exception:
                                bots = None
                                self.chat_id = None
                                self.token = None
                                logging.info("Token seems to be invalid (or internet access is restricted)!")

            if bots is None or self.chat_id is None or self.token is None:
                logging.info("Could not set up telegram bot for some reason !")

    @staticmethod
    def transmit_message(chat_id, token, message):
        success = False
        try:
            from telegrambotapiwrapper import Api
            bot = Api(token=token)
            bot.send_message(chat_id=chat_id, text=message)
            success = True
        except Exception:
            pass
        return success

    def flush(self):
        if self.chat_id is not None and self.token is not None:
            if len(self.message) > 0:
                self.transmit_message(
                    chat_id=self.chat_id,
                    token=self.token,
                    message=self.message)

    def sendmessage(self, message, *args):
        if self.chat_id is not None and self.token is not None:
            elapsed = time.time() - self.last_time
            message = message % args
            if len(self.message) == 0:
                self.message = message
            else:
                self.message += "\n%s" % message
            if elapsed >= self.flush_secs:
                success = self.transmit_message(
                    chat_id=self.chat_id,
                    token=self.token,
                    message=self.message)
                if success:
                    self.message = ""
                self.last_time = time.time()
