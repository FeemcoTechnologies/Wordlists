# Wordlists
Just a simple wordlists repo.

## BadPasswords
this list is designed to showcase passwords found and known publicly that have found during testing. Tried to upload this, github wasn't having any of that. Too large. Tried to shrink, tried to adjust, still far too large. So... see pwlist-minimize section

## web-suffixes
needed to use this to find a hidden file once. More reasons why you should fuzz 403s, sometimes they didn't lock down the subfolders. 

## pwlist-minimize
Long story short, having massive wordlists is cool, being able to shrink them and expand them with some likelihood of success, is even better. So told AI do create a thing, and it's... well it's not horrible, but it'll need to be modified to be really useful. Edits to come from that.
```
pwlist_collapse.py -i /wordlists/wpa-temp2 -o /wordlists/min-test.txt  --target-coverage 0.9 --max-seeds 1000 --llm-model distilgpt2 --llm-cache-dir /tmp/llm_cache
python3 /share/pwlist_miner.py --input /wordlists/min-test.txt --output /tmp/deleteme
john --stdout --rules:best64 -w:/tmp/deleteme |hashcat /share/pcaps/output.22000 --markov-disable
```
