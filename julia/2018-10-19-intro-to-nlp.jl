using Cascadia
using Gumbo
using HTTP
using TextAnalysis

const url = "https://en.wikipedia.org/wiki/Philippines";

function htmldoc(url::String)
    return parsehtml(String(HTTP.get(url).body))
end

html = htmldoc(url);
qres = eachmatch(sel".mw-parser-output p", html.root);
crps = Corpus(String(qres[3]))


qres[7]
scraper(html)

function scraper(html::HTMLDocument)
    qres = eachmatch(sel".mw-parser-output p", html.root);
    txts = String[]
    for p in qres[3:7]
        for tag in p.children
            println(tag)
            if isa(tag, HTMLText)
                push!(txts, tag.text)
            elseif isa(tag, HTMLNode)
                if isa(tag[1], HTMLText)
                    push!(txts, tag[1].text)
                else
                    continue
                end   
            end
        end                 
    end

    return txts
end

