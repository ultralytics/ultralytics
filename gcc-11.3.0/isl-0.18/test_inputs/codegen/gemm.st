domain: "[ni, nj, nk] -> { S_4[i, j, k] : k <= -1 + nk and k >= 0 and j <= -1 + nj and j >= 0 and i <= -1 + ni and i >= 0; S_2[i, j] : j <= -1 + nj and j >= 0 and i <= -1 + ni and i >= 0 }"
child:
  set:
  - filter: "[ni, nj, nk] -> { S_4[i, j, k]; S_2[i, j] }"
    child:
      schedule: "[ni, nj, nk] -> [{ S_4[i, j, k] -> [(i)]; S_2[i, j] -> [(i)] }, { S_4[i, j, k] -> [(j)]; S_2[i, j] -> [(j)] }, { S_4[i, j, k] -> [(k)]; S_2[i, j] -> [(0)] }]"
      permutable: 1
      coincident: [ 1, 1, 0 ]
      child:
        sequence:
        - filter: "[ni, nj, nk] -> { S_2[i, j] }"
        - filter: "[ni, nj, nk] -> { S_4[i, j, k] }"
