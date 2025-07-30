{
  for (int c0 = 1; c0 <= 6; c0 += 2) {
    for (int c2 = 1; c2 <= c0; c2 += 1) {
      S1(c0, (c0 - 1) / 2, c2);
      S2(c0, (c0 - 1) / 2, c2);
    }
    for (int c2 = c0 + 1; c2 <= p; c2 += 1)
      S1(c0, (c0 - 1) / 2, c2);
  }
  for (int c0 = 7; c0 <= m; c0 += 2)
    for (int c2 = 1; c2 <= p; c2 += 1)
      S1(c0, (c0 - 1) / 2, c2);
}
