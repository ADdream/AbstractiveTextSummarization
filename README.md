# AbstractiveTextSummarization
Implementation of Abstractive Text Summarization using a seq2seq model with local attention. Used Bidirectional LSTM for encoding and LSTM for decoding. For generating context vector used predictive alignment instead of monotonic alignment.

Implemented with reference to the paper [https://nlp.stanford.edu/pubs/emnlp15_attn.pdf]. During training( not from the logic present in the paper) in the decoding layer instead of using the generated previous hidden state the actual value of the
target word is taken( i.e used Teacher Forcing).

The above logic is used so that the model will converge fastly.

While testing and validating used the actual generated previous hidden state.
