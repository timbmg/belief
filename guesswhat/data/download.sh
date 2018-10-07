cd guesswhat/data

echo "GuessWhat?! training, validation and test sets"
curl -O https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz
curl -O https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz
curl -O https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz

echo "vgg fc8 features and mapping file"
curl -O https://drive.google.com/open?id=1P43v6qFe5WkcJEuzbZpoSNK86e0uLAE6
curl -O https://drive.google.com/open?id=12QgC7XWY1iRhm-k4W8xRV1U1UgLlB37Z

echo "Vocabluary and Categories"
curl -O https://drive.google.com/open?id=1c2eEae57HRsKVW8PXBakF4wylVJ3dJbr
curl -O https://drive.google.com/open?id=17Rdau-lq2HwOYH4q9idSyEA6m4T9062e


cd ../bin
echo "Guesser, QGen and Oracle binaries"
# guesser
curl -O https://drive.google.com/open?id=1SuucKkaeKK4t90trjuZY0cdKLWijrI3G
# oralce
curl -O https://drive.google.com/open?id=164grPXd5MNpq5UhC9VN6Uq1VVuEOoWjB
# qgen
curl -O https://drive.google.com/open?id=1DLbAB1H2SNk03Wau-cYAbYZUYJyGyw10
