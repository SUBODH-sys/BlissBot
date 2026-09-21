# BlissBot
Pyscoeducation ChatBot 

### To save any changes done directly in Github
1. git add .
2. git commit -m "Your descriptive commit message"
3. $ git push origin main

### To save any changes done directly in CodeSpaces
1. git fetch origin
2. git pull

I tested BM25 + RRF and a cross-encoder reranker against dense-only retrieval. Both reduced accuracy on my golden set (Hit@1 0.886 → 0.780 and 0.837) and the reranker added ~3 s latency, so I shipped dense-only