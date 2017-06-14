######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from urlparse import urlparse, parse_qs

from utils.commandparser import NNSDSOptParser
from nn.NNDialogue import NNDial

import json    
import cgi
import ssl
import numpy as np

class ChatHandler(BaseHTTPRequestHandler):
    model = None

    def do_OPTIONS(self):
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')    
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")   
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Credentials", "false")

    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Access-Control-Allow-Credentials", "false")
        self.send_header('Content-type','application/json')  
        self.end_headers() 

        q = parse_qs(urlparse(self.path).query,\
                keep_blank_values=True)
        #print q
        response = model.reply(
                q['user_utt_t'][0], q['generated'][0],
                q['selected_venue'][0],q['venue_offered'][0],
                q['belief_t'][0]     )
        self.request.sendall(json.dumps(response))
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Access-Control-Allow-Credentials", "false")
        self.send_header('Content-type','application/json')  
        self.end_headers() 
        
        data = self.rfile.read(int(self.headers['Content-Length']))
        q = json.loads(data)
        response = model.reply(
                q['user_utt_t'],  q['generated'],int(q['selected_venue']),
                json.loads(q['venue_offered']), q['belief_t']   )
        self.wfile.write(json.dumps(response))
        self.wfile.close()
        return 

if __name__ == '__main__':

    # TODO: IP address and port
    hostname = 'xxx.xxx.xxx.xxx'
    port = ????
    
    # loading neural dialog model
    args = NNSDSOptParser()
    config = args.config
    model = NNDial(config,args)

    # handler
    ChatHandler.model = model 
    
    # launch server
    httpd = HTTPServer((hostname,port), ChatHandler)
    print 'Server is on - %s:%d' % (hostname,port)
    httpd.serve_forever()




