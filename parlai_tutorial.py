import os
# Import the Interactive script
from parlai.scripts.interactive import Interactive

# call it with particular args
Interactive.main(
    # the model_file is a filename path pointing to a particular model dump.
    # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
    # They'll be automatically downloaded when you ask to use them.
    model_file='zoo:tutorial_transformer_generator/model'
)

# The display_data script is used to show the contents of a particular task.
# By default, we show the train
from parlai.scripts.display_data import DisplayData
DisplayData.main(task='empathetic_dialogues', num_examples=5)

# we can instead ask to see fewer examples, and get them from the valid set.
DisplayData.main(task='empathetic_dialogues', num_examples=3, datatype='valid')


# we'll save it in the "from_scratch_model" directory
location = "/home/ubuntu/work/from_hrithik/ParlAI/ParlAI_Tutorial"
dir = from_scratch_model
path = os.path.join(location, dir)
shutil.rmtree(path, ignore_errors=False)
# rm -rf from_scratch_model
os.mkdir('from_scratch_model') 
#mkdir -p from_scratch_model

from parlai.scripts.train_model import TrainModel
TrainModel.main(
    # we MUST provide a filename
    model_file='from_scratch_model/model',
    # train on empathetic dialogues
    task='empathetic_dialogues',
    # limit training time to 2 minutes, and a batchsize of 16
    max_train_time=2 * 60,
    batchsize=16,
    
    # we specify the model type as seq2seq
    model='seq2seq',
    # some hyperparamter choices. We'll use attention. We could use pretrained
    # embeddings too, with embedding_type='fasttext', but they take a long
    # time to download.
    attention='dot',
    # tie the word embeddings of the encoder/decoder/softmax.
    lookuptable='all',
    # truncate text and labels at 64 tokens, for memory and time savings
    truncate=64,
)


# rm -rf from_pretrained
# mkdir -p from_pretrained
location = "/home/ubuntu/work/from_hrithik/ParlAI/ParlAI_Tutorial"
dir = from_pretrained
path = os.path.join(location, dir)
shutil.rmtree(path, ignore_errors=False)
os.mkdir('from_pretrained') 

TrainModel.main(
    # similar to before
    task='empathetic_dialogues', 
    model='transformer/generator',
    model_file='from_pretrained/model',
    
    # initialize with a pretrained model
    init_model='zoo:tutorial_transformer_generator/model',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
    label_truncate=128, ffn_size=2048, embedding_size=512,
    activation='gelu', variant='xlm',
    dict_lower=True, dict_tokenizer='bpe',
    dict_file='zoo:tutorial_transformer_generator/model.dict',
    learn_positional_embeddings=True,
    
    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=1e-5, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='ppl',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=600, validation_every_n_epochs=0.25,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=12, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
)

# note that if you want to see model-specific arguments, you must specify a model name
print(TrainModel.help(model='seq2seq'))

from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
)

from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
    skip_generation=False,
)

from parlai.core.teachers import register_teacher, DialogTeacher

@register_teacher("my_teacher")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile):
        # filename tells us where to load from.
        # We'll just use some hardcoded data, but show how you could read the filename here:
        print(f" ~~ Loading from {datafile} ~~ ")
        
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        
        # first episode
        # notice how we have call, response, and then True? The True indicates this is a first message
        # in a conversation
        yield ('Hello', 'Hi'), True
        # Next we have the second turn. This time, the last element is False, indicating we're still going
        yield ('How are you', 'I am fine'), False
        yield ("Let's say goodbye", 'Goodbye!'), False
        
        # second episode. We need to have True again!
        yield ("Hey", "hi there"), True
        yield ("Deja vu?", "Deja vu!"), False
        yield ("Last chance", "This is it"), False
        
        
DisplayData.main(task="my_teacher")

DisplayModel.main(task='my_teacher', model_file='from_pretrained/model', skip_generation=False)


from parlai.core.agents import register_agent, Agent

@register_agent("hello")
class HelloAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser.add_argument('--name', type=str, default='Alice', help="The agent's name.")
        return parser
        
    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.id = 'HelloAgent'
        self.name = opt['name']
    
    def observe(self, observation):
        # Gather the last word from the other user's input
        words = observation.get('text', '').split()
        if words:
            self.last_word = words[-1]
        else:
            self.last_word = "stranger!"
    
    def act(self):
        # Always return a string like this.
        return {
            'id': self.id,
            'text': f"Hello {self.last_word}, I'm {self.name}",
        }


DisplayModel.main(task='my_teacher', model='hello')

import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga


class Encoder(nn.Module):
    """
    Example encoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size.
    Pay particular attention to the ``forward`` output.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.
        Arguments here can be used to provide hyperparameters.
        """
        # must call super on all nn.Modules.
        super().__init__()

        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, input_tokens):
        """
        Perform the forward pass for the encoder.
        Input *must* be input_tokens, which are the context tokens given
        as a matrix of lookup IDs.
        :param input_tokens:
            Input tokens as a bsz x seqlen LongTensor.
            Likely will contain padding.
        :return:
            You can return anything you like; it is will be passed verbatim
            into the decoder for conditioning. However, it should be something
            you can easily manipulate in ``reorder_encoder_states``.
            This particular implementation returns the hidden and cell states from the
            LSTM.
        """
        embedded = self.embeddings(input_tokens)
        _output, hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    """
    Basic example decoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size. Decoder allows for incremental decoding by ingesting the
    current incremental state on each forward pass.
    Pay particular note to the ``forward``.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.
        Arguments here can be used to provide hyperparameters.
        """
        super().__init__()
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, input, encoder_state, incr_state=None):
        """
        Run forward pass.
        :param input:
            The currently generated tokens from the decoder.
        :param encoder_state:
            The output from the encoder module.
        :parm incr_state:
            The previous hidden state of the decoder.
        """
        embedded = self.embeddings(input)
        if incr_state is None:
            # this is our very first call. We want to seed the LSTM with the
            # hidden state of the decoder
            state = encoder_state
        else:
            # We've generated some tokens already, so we can reuse the existing
            # decoder state
            state = incr_state

        # get the new output and decoder incremental state
        output, incr_state = self.lstm(embedded, state)

        return output, incr_state


class ExampleModel(tga.TorchGeneratorModel):
    """
    ExampleModel implements the abstract methods of TorchGeneratorModel to define how to
    re-order encoder states and decoder incremental states.
    It also instantiates the embedding table, encoder, and decoder, and defines the
    final output layer.
    """

    def __init__(self, dictionary, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)

    def output(self, decoder_output):
        """
        Perform the final output -> logits transformation.
        """
        return F.linear(decoder_output, self.embeddings.weight)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states to select only the given batch indices.
        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        """
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        """
        Reorder the decoder states to select only the given batch indices.
        This method can be a stub which always returns None; this will result in the
        decoder doing a complete forward pass for every single token, making generation
        O(n^2). However, if any state can be cached, then this method should be
        implemented to reduce the generation complexity to O(n).
        """
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]


@register_agent("my_first_lstm")
class Seq2seqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.
    Implements the interface for TorchGeneratorAgent. The minimum requirement is that it
    implements ``build_model``, but we will want to include additional command line
    parameters.
    """

    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt):
        """
        Add CLI arguments.
        """
        # Make sure to add all of TorchGeneratorAgent's arguments
        super().add_cmdline_args(argparser)

        # Add custom arguments only for this model.
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=1024, help='Hidden size.'
        )

    def build_model(self):
        """
        Construct the model.
        """

        model = ExampleModel(self.dict, self.opt['hidden_size'])
        # Optionally initialize pre-trained embeddings by copying them from another
        # source: GloVe, fastText, etc.
        self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

# of course, we can train the model! Let's Train it on our silly toy task from above
rm -rf my_first_lstm
mkdir -p my_first_lstm

location = "/home/ubuntu/work/from_hrithik/ParlAI/ParlAI_Tutorial"
dir = my_first_lstm
path = os.path.join(location, dir)
shutil.rmtree(path, ignore_errors=False)
os.mkdir('my_first_lstm') 

TrainModel.main(
    model='my_first_lstm',
    model_file='my_first_lstm/model',
    task='my_teacher',
    batchsize=1,
    validation_every_n_secs=10,
    max_train_time=60,
)

DisplayModel.main(model_file='my_first_lstm/model', task='my_teacher')

