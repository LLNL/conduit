
/*
dtypes:

int8,byte
int16,short
int32,int
int64

uint8,ubyte
uint16,ushort
unit32,uint
uint64

float16,half
float32,single,float
float64,double

For the hackaton, we will implement :

uint32  {uint}
uint64
float64 {double}

DArray: (base_darray)
 void      *m_ptr;     // data 
 dtype      m_dtype;   // data type (for now: {uint32,uint64,float64})
 uint64     m_offset;  //  bytes from m_ptr to start of array
 uint64     m_stride;  //  bytes between start of current and start of next
 uint64     m_size;    // number of entries
 bool       m_alloced; // do we own the memory

base_darray(const std::string &dtype_name, uint64 size=1, {etc}); // results in alloced
base_darray(dtype dt, uint64 size=1, {etc}); // results in alloced
base_darray(void *data, uint64 size, {etc}); 

{etc} = uint64 offset=0, uint64 stride=0)

Other things we need but may not need to touch for the hackathon:
 endianness m_endian;  // {big,little,default}
 uint64     m_pad;     // unused space after data in bytes


Objects:
DArray
DType
List
Node

Schema
SchemaRegistry


A node can:
 hold keys + values (dtype == object) use std methods
 be a List (dtype == list) [as_list]
 be a DArray (dtype == various darray types) [as_{darray}]
 
Node methods:

uint32      as_uint32() const
uint32     *as_uint32_ptr()
uint32      to_uint32() const 
void        to_uint32_darray(uint32 *) const // ptr is already alloced
void        to_uint32_darray(uint32_darray &) const // darray is already alloced to match
void        to_uint32_darray(vector<uint32> &) const

float64     as_float64() const
float64    *as_float64_ptr()
float64     to_float64() const
void        to_float64_darray(float64 *) const // alloced 
void        to_uint32_darray(float64_darray &) const // darray is already alloced to match
void        to_float64_darray(vector<float64> &) const

DType dtype() const;

bool has_key(const std::string &name) const;
void keys(std::vector<std::string> &keys) const;

Node &value(const std::string &key_name);

const Node &value(const std::string &key_name) const;
Node &operator[](const std::string &key_name);
const Node &operator[](const std::string &key_name) const;

*/

int main()
{
    conduit::register_schemas(schemas); // std::string, char *, uchar *, ifstream &
    conduit::uint32 *data_ptr = read_data();
    conduit::Node root(data_ptr,"domain");

    cout << root["id"].dtype().to_string() << endl; // "uint32"
    
    cout << root["id"].as_unit32() << endl;

    
    conduit::uint32        *lext_ptr = root["logical_extents"].as_unit32_ptr();
    // or
    conduit::unit32_tuple  &lext     = root["logical_extents"].as_unit32_tuple();

    //  print the contents of the tuple
    cout << lext.to_string() <<endl;

    conduit::List &nei_list = root["neighbors"];

    cout << "# of neighbors " << nei_list.length() << endl;

    for(int i=0;i<nei_list.length();++i)
    {
        conduit::Node nei = nei_list[i];
        cout << nei["id"].to_float64() <<endl;
        cout << nei["orientation"].to_string() <<endl;
        // we will actually overload stream redirect
        cout << nei["overlap"].as_unit32_tuple() << endl;
    
    }

}


