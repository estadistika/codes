using Cascadia
using DataFrames
using Gumbo
using HTTP

url = "https://www.phivolcs.dost.gov.ph/html/update_SOEPD/EQLatest-Monthly/2018/2018_September.html";
res = HTTP.get(url);

body = String(res.body);
html = parsehtml(body);

qres = eachmatch(sel".auto-style33", html.root);

for elem in qres
    println(replace(text(elem[1]), r"\n|\t|\s" => s""))
end

# access all HTML DOM with .MsoNormalTable class
qdat = eachmatch(sel".MsoNormalTable", html.root);

# number of HTML DOM with .MsoNormalTable class
length(qdat)

# september table is at 3rd index
table = qdat[3];

# access the table body
tbody = table[1];

# check the number of rows
length(tbody.children)

# access the first row
tbody[3]

# 3 and 4 contains the october data
# check number of rows
length(qdat[3][1].children)

# drill down: table > tbody > tr (3rd row)
qdat[3][1][3]

# drill down: table > tbody > tr (3rd row) > td (1st cell)
qdat[3][1][3][1]

# check children
qdat[3][1][3][1].children

# drill down: table > tbody > tr (3rd row) > td (1st cell) > span (3rd element)
qdat[3][1][3][1][3]

# drill down: table > tbody > tr (3rd row) > td (1st cell) > span (3rd element) > a > span
qdat[3][1][3][1][3][1][1]

# drill down: table > tbody > tr (3rd row) > td (1st cell) > span (3rd element) > a > span > text
qdat[3][1][3][1][3][1][1][1].text

# remove leading and trailing whitespaces
strip(qdat[3][1][3][1][3][1][1][1].text)

strip(qdat[3][1][3][2][1].text)




date, locn = String[], String[]
latd, lond, dept, magn = Float64[], Float64[], Float64[], Float64[]

tbody = qdat[3][1];
@time for tr in tbody.children[3:end]
    counter, data = 1, Any[]
    for td in tr.children
        if counter == 1
            try
                if length(td[1].children) == 2
                    if length(td[1][2][1].children) == 2
                        push!(data, replace(strip(td[1][1].text) * " " * strip(td[1][2][1][2].text), r"\n|\t" => s""))
                    else
                        if strip(td[1][1].text) == "0"
                            push!(data, replace(strip(td[1][1].text) * "" * strip(td[1][2][1][1].text), r"\n|\t" => s""))
                        else
                            push!(data, replace(strip(td[1][1].text) * " " * strip(td[1][2][1][1].text), r"\n|\t" => s""))
                        end
                    end
                else
                    if !isa(td[1][1][1], HTMLText)
                        try
                            if length(td[1][1].children) == 2
                                if !isa(td[1][1][2], HTMLText)
                                    push!(data, replace(strip(td[1][1][2][1].text), r"\n|\t" => s""))
                                else
                                    push!(data, replace(strip(td[1][1][2].text), r"\n|\t" => s""))
                                end
                            else
                                if length(td[1][1][1].children) == 2
                                    push!(data, replace(strip(td[1][1][1][2].text), r"\n|\t" => s""))
                                else
                                    push!(data, replace(strip(td[1][1][1][1].text), r"\n|\t" => s""))
                                end
                            end
                        catch
                            push!(data, replace(strip(td[1][1][1][1][1].text), r"\n|\t" => s""))
                        end
                    else
                        push!(data, replace(strip(td[1][1][1].text), r"\n|\t" => s""))
                    end
                end
            catch
                if !isa(td[2][1][1], HTMLText)
                    push!(data, replace(strip(td[1].text) * " " * strip(td[2][1][1][1].text), r"\n|\t" => s""))
                else
                    push!(data, replace(strip(td[1][1].text) * " " * strip(td[2][1][1].text), r"\n|\t" => s""))
                end
            end
        else
            try
                if match(r".$", td[1].text) == nothing
                    push!(data, strip(td[1].text))
                else
                    push!(data, replace(strip(td[1].text), r".$" => s""))
                end
            catch
                push!(data, strip(td[1][1].text))
            end
        end
        counter += 1
    end
    
    push!(date, convert(String, data[1]))
    push!(locn, convert(String, data[6]))

    push!(latd, parse(Float64, data[2]))
    push!(lond, parse(Float64, data[3]))
    push!(dept, parse(Float64, data[4]))
    push!(magn, parse(Float64, data[5]))
end

pvdat = DataFrame(
    Date = date,
    Latitude = latd,
    Longitude = lond,
    Depth = dept,
    Magnitude = magn,
    Location = locn
)

length(dept)
#=
Wrapping into Functions
=#

function htmldoc(url::String)
    return parsehtml(String(HTTP.get(url).body))
end

function header(html::HTMLDocument)
    qres = eachmatch(sel".auto-style33", html.root)

    txts = String[];
    for elem in qres
        push!(txts, replace(text(elem[1]), r"\n|\t|\s" => s""))
    end

    return txts
end

function scraper(html::Array{HTMLNode,1})
    date, locn = String[], String[]
    latd, lond, dept, magn = Float64[], Float64[], Float64[], Float64[]

    tbody = html[3][1];
    for tr in tbody.children[3:end]
        counter, data = 1, Any[]
        for td in tr.children
            if counter == 1
                try
                    data = firstcolumn(td, data)
                catch
                    data = firstcolumn(td, data, :catch)
                end
            else
                try
                    if match(r"\.$", td[1].text) == nothing
                        push!(data, strip(td[1].text))
                    else
                        push!(data, replace(strip(td[1].text), r"\.$" => s""))
                    end
                catch
                    push!(data, strip(td[1][1].text))
                end
            end
            counter += 1
        end
        
        push!(date, convert(String, data[1]))
        push!(locn, convert(String, data[6]))

        push!(latd, parse(Float64, data[2]))
        push!(lond, parse(Float64, data[3]))
        push!(dept, parse(Float64, data[4]))
        push!(magn, parse(Float64, data[5]))
    end

    df = DataFrame(
        Date = date,
        Latitude = latd,
        Longitude = lond,
        Depth = dept,
        Magnitude = magn,
        Location = locn
    )

    return df
end

function firstcolumn(html::HTMLElement{:td}, data::Array{Any, 1}, section::Symbol = :try)
    if section == :try
        if length(html[1].children) == 2
            if length(html[1][2][1].children) == 2
                push!(data, replace(strip(html[1][1].text) * " " * strip(html[1][2][1][2].text), r"\n|\t" => s""))
            else
                if strip(html[1][1].text) == "0"
                    push!(data, replace(strip(html[1][1].text) * "" * strip(html[1][2][1][1].text), r"\n|\t" => s""))
                else
                    push!(data, replace(strip(html[1][1].text) * " " * strip(html[1][2][1][1].text), r"\n|\t" => s""))
                end
            end
        else
            if !isa(html[1][1][1], HTMLText)
                try
                    if length(html[1][1].children) == 2
                        if !isa(html[1][1][2], HTMLText)
                            push!(data, replace(strip(html[1][1][2][1].text), r"\n|\t" => s""))
                        else
                            push!(data, replace(strip(html[1][1][2].text), r"\n|\t" => s""))
                        end
                    else
                        if length(html[1][1][1].children) == 2
                            push!(data, replace(strip(html[1][1][1][2].text), r"\n|\t" => s""))
                        else
                            push!(data, replace(strip(html[1][1][1][1].text), r"\n|\t" => s""))
                        end
                    end
                catch
                    push!(data, replace(strip(html[1][1][1][1][1].text), r"\n|\t" => s""))
                end
            else
                push!(data, replace(strip(html[1][1][1].text), r"\n|\t" => s""))
            end
        end
    elseif section == :catch
        if !isa(html[2][1][1], HTMLText)
            push!(data, replace(strip(html[1].text) * " " * strip(html[2][1][1][1].text), r"\n|\t" => s""))
        else
            push!(data, replace(strip(html[1][1].text) * " " * strip(html[2][1][1].text), r"\n|\t" => s""))
        end
    else
        error("section takes either: :try or :catch.")
    end

    return data
end

addr = "https://www.phivolcs.dost.gov.ph/html/update_SOEPD/EQLatest-Monthly/2018/2018_September.html"
html = htmldoc(addr);
qdat = eachmatch(sel".MsoNormalTable", html.root);
data = scraper(qdat);
data