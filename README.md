pesubmit
========

A small Python tool to submit [Project Euler](http://projecteuler.net) solutions (by logging in, solving the captcha, and submitting a http post with the provided solution).

Usage (from the command line):

    python3 pesubmit.py 1 987654321

or (after you've given it execute permission, `chmod +x pesubmit.py`) just 

    /.pesubmit.py 1 987654321

where `1` is the problem number, and `987654321` is your solution (and no, it's not the correct solution...).

Usage (from Python itself):

    import pesubmit
    ...
    pesubmit.submit(1,987654321)
    

```
Fabians-MBA-15:pesubmit frl$ ./pesubmit.py 1 987654321
2013-12-02 06:02:54,950 INFO: Submitting. Problem: 1, Soln: 987654321
2013-12-02 06:02:57,808 INFO: Login succeeded, logged in as fabianlischka
2013-12-02 06:02:58,141 WARNING: Already Completed on Fri, 18 Jan 2013, 20:49
```

Requirements
============

You'll need Python 3, and these modules installed: bs4, keyring, png. You can get them by running

    pip3 install BeautifulSoup4 keyring pypng

You'll only need pesubmit.py (not pesubmit-utils.py).
