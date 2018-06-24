using JuliaDB
using JuliaDBMeta

data_url = "https://raw.githubusercontent.com/estadistika/assets/master/data/nycflights13.csv";
down_dir = joinpath(homedir(), "Downloads", "nycflights13.csv");
download(data_url, down_dir)

# load the csv file
nycflights = loadtable(down_dir);

# filter the column
cols = (:year, :month, :day, :dep_time, :dep_delay, :arr_time, :arr_delay, :tailnum, :air_time, :dest, :distance);
flights = select(nycflights, cols);

# Access the dimension of the data
length(rows(flights))
length(columns(flights))

# Accessing the Head and Tail of the Data
select(flights, cols)[1:6]
select(flights, cols)[end - 5:end]

"""
Filter rows with `filter()` or `@filter`
"""
# using JuliaDB
filter((:month => x -> x .== 1, :day => x -> x .== 1), flights); # remove ; to see result

# using JuliaDBMeta
@filter flights :month .== 1 && :day .== 1

"""
Arrange rows with `Base.sort`
"""
sort(flights, (:year, :month, :day))
sort(flights, (:year, :month, :day), rev = true)

"""
Select rows with `select`
"""
select(flights, (:year, :month, :day))

# select columns between year and day (inclusive)
select(flights, Between(:year, :day))

# select all columns except those from year to day (inclusive)
select(flights, Not(Between(:year, :day)))

"""
Rename column using renamecol
"""
renamecol(flights, :tailnum, :tail_num)

"""
Add new column with `insetcol`, `insertcolafter`, `insertcolbefore`, and @transform
"""

gain = map(x -> x.arr_delay - x.dep_delay, flights, select = (:arr_delay, :dep_delay));
speed = map(x -> x.distance / x.air_time * 60, flights, select = (:distance, :air_time));
insertcolafter(flights, :distance, :gain, gain)
insertcolafter(flights, :distance, :speed, speed)

# Using JuliaDBMeta
@transform flights {gain = :arr_delay - :dep_delay, speed = :distance / :air_time * 60}

"""
Summarise values with summarize()
"""
summarize(mean, dropna(flights), select = :dep_delay)
@with dropna(flights) mean(:dep_delay)

"""
Grouped operations
"""
# Using JuliaDB
delay = groupby(
    @NT(
        count = length, 
        dist = :distance => x -> mean(dropna(x)), 
        delay = :arr_delay => x -> mean(dropna(x))
        ), 
        flights, 
        :tailnum
);

# Using JuliaDBMeta
delay = @groupby flights :tailnum {
    count = length(_),
    dist = mean(dropna(:distance)),
    delay = mean(dropna(:arr_delay))
}

delay = filter((:count => x -> x .> 20, :dist => x -> x .< 2000), delay)

using Gadfly
Gadfly.push_theme(:dark)
using IterableTables

p = plot(
    filter(:delay => x -> !isnan(x), delay), 
    layer(
        x = :dist, 
        y = :delay,
        Geom.smooth,
        style(default_color = colorant"red", line_width = 2pt)
    ),
    layer(        
        x = :dist, 
        y = :delay,
        color = :count,
        size = :count,
        Geom.point,
        style(default_color = colorant"orange", highlight_width = 0pt)
    )
)

# dimension in golden ratio
draw(PNG("2018-6-8-p1.png", 7.28115inch, 4.5inch), p)

destinations = groupby(
    @NT(
        planes = :tailnum => x -> length(unique(x)),
        flights = length
    ),
    flights,
    :dest
);

destinations = @groupby flights :dest {
    planes = length(unique(:tailnum)),
    flights = length(_)
}

group = (:year, :month, :day)
per_day = groupby(
    @NT(flights = length),
    flights, 
    group
)

per_month = groupby(
    @NT(flights = :flights => x -> sum(x)),
    select(per_day, Not(:day)), 
    group[1:2]
)

per_year = groupby(
    @NT(flights = :flights => x -> sum(x)),
    select(per_month, Not(:month)), 
    group[1]
)
 
"""
Selecting operation
"""
select(flights, :year)
select(flights, 1)

@apply dropna(flights) begin
    @groupby (:year, :month, :day) {
        arr = mean(:arr_delay),
        dep = mean(:dep_delay)
    }
    @filter :arr .> 30 || :dep .> 30
end

@time 
nycflights = loadtable(down_dir);
save(nycflights, joinpath(homedir(), "Downloads", "d"))