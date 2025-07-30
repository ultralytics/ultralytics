{
  for (int c0 = 2; c0 <= 3; c0 += 1)
    for (int c1 = -c0 + 6; c1 <= 6; c1 += 1)
      S1(c0, c1);
  for (int c0 = 4; c0 <= 8; c0 += 1) {
    if (c0 >= 6) {
      S2(c0, -c0 + 9);
    } else {
      if (c0 == 4)
        for (int c1 = 3; c1 <= 4; c1 += 1)
          S1(4, c1);
      S1(c0, -c0 + 9);
      S2(c0, -c0 + 9);
    }
    for (int c1 = max(c0 - 1, -c0 + 10); c1 <= 6; c1 += 1)
      S1(c0, c1);
  }
}
