## One Model for All Stocks

What happens if training one algorithmic trading algorithm for all stocks?

### Advantage:
* Technical analysis indicators along with other features should generally have same pattern
* No need to specifically train model for each stock

### Disadvantage:
* More costly to train: forfeit one of the advantage of TFJ-DRL
* Being a general algorithm, it loses the ability to customize for one stock
* Performance varies across different stocks. Some performs extremely well (2x profit), while some extremely bad (loss in hundreds). Others have mixed performance. How can you use a model when you don't trust it?

Still, it doesn't hurt to try. 

