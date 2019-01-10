using Clustering
using Dates
using Languages
using LinearAlgebra: norm
using TextAnalysis

const path = "/Users/alahmadgaidasaad/Documents/Projects/Learn-Islam/learn-islam/learn-islam.github.io/assets/quran/en.sahih.txt";
io = open(path, "r");
equran = read(io, String);

# ------
# Text Cleaning
# ------

# replace newline with space
equran = replace(equran, "\n" => " ");

# remove the verse numbers
equran = replace(equran, r"\d*\|\d*\|" => "");

equran[1:435]

# Tokenize Sentences
sentences = TextAnalysis.sentence_tokenize(Languages.English(), equran);
sentences[end] = split(sentences[end], r"\\\"")[1];

# Join the sentences into single text
cquran = join(sentences, " ");

# ------
# Text Corpus
# ------

# Define the metadata of the text
docmeta = TextAnalysis.DocumentMetadata(Languages.English(), "Al-Qur'an", "", "November 30, 2018");

# Create a string document
doc = map(x -> StringDocument(x, docmeta), sentences)
# typeof(doc)
# doc = StringDocument(cquran, docmeta);

# From this document create a corpus
crps = Corpus(doc);

# Copy the corpus
orig_crps = deepcopy(crps);

# Extract the first element of the corpus
d = orig_crps[1];
typeof(d)

# get the tokens from this document
tokens(d)

# Create NGrams
# (read more about ngrams)
ngrams(d, 2)

# Data Cleaning 
# remove non-letters, punctuation, and stopwords
prepare!(crps, strip_non_letters | strip_punctuation | strip_case | strip_stopwords)
crps

# Stem the words (cutting the word into its root word)
stem!(crps)
crps 

# update lexicon
update_lexicon!(crps) # counts the frequency of the words
update_inverse_index!(crps) # creates an index for the word present in how many documents in the corpus

# for example
crps["muhammad"]

# Document Term Matrix
# a matrix that tells which word exist in which document, and how many times it exist.
m = DocumentTermMatrix(crps);

# Extract the raw data. This is a sparse matrix
dt = dtm(m);
size(dt)

# TD-IDF
# Term Document - Inverse Document Frequency
# a word existing in one document could be a common word
# which does not mean anything. If a word exist, which word 
# mean something and which word does not mean something
tfidf = tf_idf(m);
tfidf

# use clustering
# this is a problem since the corpus contains only one document
clust = kmeans(Matrix(tfidf'), 5)
clust.counts

lexicon(crps)

# Text Summarization
c = deepcopy(orig_crps);

prepare!(c, strip_non_letters | strip_punctuation | strip_case | strip_stopwords);
# stem!(c);
update_lexicon!(c);
update_inverse_index!(c);

ctfidf = tf_idf(DocumentTermMatrix(c));
typeof(ctfidf)

# Compute the similarity between matrices
A = ctfidf * ctfidf'

# Define the pagerank
import Base: getindex;
function getindex(crps::Corpus{StringDocument{SubString{String}}}, personalization::Array{String, 1})

    out = zeros(Bool, length(personalization), length(crps)); k = 1
    for i in personalization
        l = 1
        for j in crps
            out[k, l] = issubset(i, text(j))
            l += 1
        end
        k += 1
    end

    return findall(x -> x == true, vec(map(x -> x > 0, mapslices(sum, out, dims = 1))))
end
using SparseArrays

function pagerank(crps::Corpus{StringDocument{SubString{String}}}; Niter::Int64 = 20, damping::Float64 = .15, personalization::Union{Nothing, Array{String, 1}} = nothing) 
    tfidf = tf_idf(DocumentTermMatrix(crps));

    if personalization == nothing
        return pagerank(tfidf * tfidf', Niter, damping, nothing) 
    else 
        return pagerank(tfidf * tfidf', Niter, damping, getindex(crps, personalization))
    end
end

function pagerank(A::SparseArrays.SparseMatrixCSC{Float64,Int64}, Niter::Int64, damping::Float64, personalization::Union{Nothing, Array{Int64, 1}})
    Nmax = size(A, 1)

    r = rand(1, Nmax);                   # Generate a random starting rank
    println(r)
    if personalization != nothing
        println(personalization)
        r[personalization] = 1
        println("You are here!")
    end
    
    r = r ./ norm(r, 1);                 # Normalize it
    a = (1 - damping) ./ Nmax;           # Create a damping factor

    for i = 1:Niter
        s = r * A
        s .*= damping
        r = s .+ (a * sum(r, dims = 2))  # compute pagerank
    end

    r = r ./ norm(r, 1)

    return r
end

# Define the personalized page rank
# findall(x -> x == true, extract_idx2(crps, personalization = ["mercy"]))


p = pagerank(crps)
p = pagerank(crps, personalization = ["mercy"])
p = pagerank(ctfidf * ctfidf', Niter = 30)

TextAnalysis.sentence_tokenize(Languages.English(), cquran)[sortperm(vec(p), rev = true)][1:200]
unique(TextAnalysis.sentence_tokenize(Languages.English(), cquran)[sortperm(vec(p), rev = true)][1:200])

summarize(orig_crps[1])

# Embeddings
using Embeddings

embeddings = load_embeddings(Word2Vec);
tk = tokens(orig_crps[1])
emb_ind = [findfirst(isequal(x), embeddings.vocab) for x in tk];
filter!(x -> x != nothing, emb_ind)

# using this feature, we can use this for deep learning modelling
features = embeddings.embeddings[:, emb_ind];

