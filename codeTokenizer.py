import sctokenizer

tokens = sctokenizer.tokenize_file(filepath='./dataset/tbccd_bigclonebenchdata_change/tbccd_bigclonebenchdata_change/a74.java', lang='java')
for token in tokens:
    print(token)