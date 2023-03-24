def load_pageTypes(query):
    if query == "eu":
        return ["Blogs", "Forums", "Instagram", "News", "Reddit", "Review", "Tumblr", "Twitter", "YouTube"]
    # review only for english
    else:
        return ["Blogs", "Facebook", "Forums", "Instagram", "News", "Reddit", "Tumblr", "Twitter", "YouTube"]
