///
/// file: Utils.cpp
///

#include "Utils.h"

namespace conduit
{

namespace utils
{

///============================================
void     
split_path(const std::string &path,
           std::string &curr,
           std::string &next)
{
    curr.clear();
    next.clear();
    std::size_t found = path.find("/");
    if (found != std::string::npos)
    {
        curr = path.substr(0,found);
        if(found != path.size()-1)
            next = path.substr(found+1,path.size()-(found-1));
    }
    else
    {
        curr = path;
    }
}

bool check_word_char(const char v)
{
    bool res = ( ( 'A' <= v) && 
                 (  v  <= 'Z') );
    res = res || ( ( 'a' <= v) && 
                 (  v  <= 'z') );
    res = res || v == '_';
    //std::cout << "check " << v << " == " << res << std::endl;
    return res;
}

bool check_num_char(const char v)
{
    bool res = ( ( '0' <= v) && 
                 (  v  <= '9') );
    return res;
}


///============================================
std::string
json_sanitize(const std::string &json)
{
    ///
    /// Really wanted to use regexs to solve this
    /// but posix regs are greedy & it was hard for me to construct
    /// a viable regex, vs those that support non-greedy (Python + Perl style regex)
    /// 
    /// Here are regexs I was able to us in python:
    //  *comments*
    //     Remove '//' to end of line
    //     regex: \/\/.*\?n
    //  *limited quoteless*
    //    find words not surrounded by quotes
    //    regex: (?<!"|\w)(\w+)(?!"|\w)
    //    and add quotes
    
    //
    // for now, we use a simple char by char parser
    //

    std::string res;
    bool        in_comment=false;
    bool        in_string=false;
    bool        in_id =false;
    std::string cur_id = "";
    
    for(size_t i = 0; i < json.size(); ++i) 
    {
        bool emit = true;
        // check for start & end of a string
        if(json[i] == '\"' &&  ( i > 0 && ( json[i-1] != '\\' )))
        {
            if(in_string)
                in_string = false;
            else
                in_string = true;
        }
        
        // handle two cases were we want to sanitize:
        // comments '//' to end of line & unquoted ids
        if(!in_string)
        {
            if(!in_comment)
            {
                if( json[i] == '/'  && 
                    i < (json.size()-1) && 
                    json[i+1] == '/')
                {
                    in_comment = true;
                    emit = false;
                }
            }
            
            if(!in_comment)
            {
                
                if( !in_id && check_word_char(json[i]))
                {
                    in_id = true;
                    // accum id chars
                    cur_id += json[i];
                    emit = false;
                }
                else if(in_id) // finish the id
                {
                    if(check_word_char(json[i]) || check_num_char(json[i]))
                    {
                        in_id = true;
                        // accum id chars
                        cur_id += json[i];
                        emit = false; 
                    }
                    else
                    {
                        in_id = false;
                        /// emit cur_id
                        res += "\"" + cur_id + "\"";
                        cur_id = "";
                    }
                    // we will also emit this char
                }
            }
            
            if(in_comment)
            {
                emit = false;
                if(json[i] == '\n')
                {
                    in_comment = false;
                }
            }
        }
        
        if(emit)
            res += json[i];
    }

    return res;
}
    
void indent(std::ostringstream &oss,
            index_t indent,
            index_t depth,
            const std::string &pad)
{
    for(index_t i=0;i<depth;i++)
    {
        for(index_t j=0;j<indent;j++)
        {
            oss << pad;
        }
    }
}
    
}
}