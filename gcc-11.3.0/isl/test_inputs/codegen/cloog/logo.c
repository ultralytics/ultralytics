{
  for (int c1 = 0; c1 <= 7; c1 += 1)
    S1(1, c1);
  for (int c0 = 2; c0 <= 6; c0 += 1) {
    for (int c1 = 0; c1 < c0 - 1; c1 += 1)
      S2(c0, c1);
    for (int c1 = c0 - 1; c1 <= 4; c1 += 1) {
      S1(c0, c1);
      S2(c0, c1);
    }
    for (int c1 = 5; c1 <= 7; c1 += 1)
      S1(c0, c1);
  }
  for (int c0 = 7; c0 <= 8; c0 += 1)
    for (int c1 = c0 - 1; c1 <= 7; c1 += 1)
      S1(c0, c1);
}
