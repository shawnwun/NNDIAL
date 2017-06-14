######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
"""
Have a text level dialog with any config loaded model.
"""

class Interact(object):

    def __init__(self):
        pass

    def prompt(self):
        #----- user turn
        userTurn = self.collect_user_turn()
        return userTurn

    def collect_user_turn(self):
        return raw_input("[User]:\t\t")   #TODO - fix this to be robust to junk input

    def quit(self, userTurn):
        if  userTurn=='</s> thank you goodbye </s>' or\
            userTurn=='end' or userTurn=='quit':
            return True
        return False


#END OF FILE
