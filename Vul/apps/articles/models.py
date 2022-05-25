import datetime
from django.db import models
from django.utils import timezone

# Create your models here.
class Article(models.Model):
    article_title = models.CharField('Name of article', max_length=200)
    article_text = models.TextField('Text of article')
    pub_date = models.DateTimeField('Date of publication')

    def __str__(self):
        return self.article_title

    def published_recently(self):
        return self.pub_date >= (timezone.now() - datetime.timedelta(days = 7))


class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete = models.CASCADE)
    author_name = models.CharField('Name of author', max_length=50)
    comment_text = models.CharField('Text of comment', max_length=200)

    def __str__(self):
        return self.author_name
