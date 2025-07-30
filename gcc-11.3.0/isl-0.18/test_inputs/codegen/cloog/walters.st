domain: "{ S2[i, div36, div37, div38] : 3div37 = 2 + i and i >= 1 and i <= 10 and 3div36 >= -2 + i and 3div38 <= 1 + i and 3div38 >= -1 + i and 3div36 <= i; S4[i, div36, div37, div38] : i >= 1 and i <= 10 and 3div36 <= i and 3div36 >= -2 + i and 3div37 <= 2 + i and 3div37 >= i and 3div38 <= 1 + i and 3div38 >= -1 + i; S1[i, div36, div37, div38] : 3div36 = i and i >= 3 and i <= 10 and 3div37 >= i and 3div38 <= 1 + i and 3div37 <= 2 + i and 3div38 >= -1 + i; S3[i, div36, div37, div38] : 3div38 = 1 + i and i <= 10 and i >= 2 and 3div36 >= -2 + i and 3div37 <= 2 + i and 3div36 <= i and 3div37 >= i }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i, div36, div37, div38] -> [(i)]; S4[i, div36, div37, div38] -> [(i)]; S3[i, div36, div37, div38] -> [(i)]; S2[i, div36, div37, div38] -> [(i)] }, { S1[i, div36, div37, div38] -> [(div36)]; S4[i, div36, div37, div38] -> [(div36)]; S3[i, div36, div37, div38] -> [(div36)]; S2[i, div36, div37, div38] -> [(div36)] }, { S1[i, div36, div37, div38] -> [(div37)]; S4[i, div36, div37, div38] -> [(div37)]; S3[i, div36, div37, div38] -> [(div37)]; S2[i, div36, div37, div38] -> [(div37)] }, { S1[i, div36, div37, div38] -> [(div38)]; S4[i, div36, div37, div38] -> [(div38)]; S3[i, div36, div37, div38] -> [(div38)]; S2[i, div36, div37, div38] -> [(div38)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[i, div36, div37, div38] }"
      - filter: "{ S2[i, div36, div37, div38] }"
      - filter: "{ S3[i, div36, div37, div38] }"
      - filter: "{ S4[i, div36, div37, div38] }"
