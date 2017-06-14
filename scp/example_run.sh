######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

TRACKER="CamRest.tracker.model"

###################################################################
######################### Train Trackers ##########################
###################################################################
# Run the tracker training first
THEANO_FLAGS="optimizer=fast_compile" python nndial.py -config config/tracker.cfg -mode train

###################################################################
########################## Valinna NDM ############################
###################################################################
# Then copy the pretrained trackers and continue to train the other parts
cp model/$TRACKER model/CamRest.NDM.model
THEANO_FLAGS="optimizer=fast_compile" python nndial.py -config config/NDM.cfg -mode adjust

###################################################################
###################### Attention-based NDM ########################
###################################################################
# Train the attention-based NDM
cp model/$TRACKER model/CamRest.Att-NDM.model
THEANO_FLAGS="optimizer=fast_compile" python nndial.py -config config/Att-NDM.cfg -mode adjust

###################################################################
############################## LIDM ###############################
###################################################################
# Train the LIDM model
cp model/$TRACKER model/CamRest.LIDM.model
THEANO_FLAGS="optimizer=fast_compile" python nndial.py -config config/LIDM.cfg -mode adjust

# Fine-tune policy network by RL
cp model/CamRest.LIDM.model model/CamRest.LIDM-RL.model
THEANO_FLAGS="optimizer=fast_compile" python nndial.py -config config/LIDM-RL.cfg -mode rl

###################################################################
################## Validation and Testing #########################
###################################################################
# Now you can test or validate the model
python nndial.py -config config/NDM.cfg -mode valid
python nndial.py -config config/NDM.cfg -mode test

# Or you can interact with it
python nndial.py -config config/NDM.cfg -mode interact


