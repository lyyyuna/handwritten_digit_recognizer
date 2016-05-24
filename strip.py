def delblankline(infile, outfile):
    """ Delete blanklines of infile """
    infp = open(infile, "r")
    outfp = open(outfile, "w")

    ss = infp.read()
    nss = ss.replace('\r\n', '\n')
    outfp.write(nss)

    infp.close()
    outfp.close()

if __name__ == "__main__":
    delblankline("output.csv","submit0503_sb.csv")
